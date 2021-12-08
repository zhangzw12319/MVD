'''
MVD(Multimodal Variational Distillation) Mindspore version
DATE:2021.11
Developer List:
[@zhangzw12319](https://github.com/zhangzw12319)
'''


import os
import os.path as osp
import argparse
import time
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_trans

from mindspore import context, load_checkpoint, \
    load_param_into_net, save_checkpoint, DatasetHelper
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_group_size
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.train.callback import LossMonitor
from mindspore.nn import SGD, Adam

from PIL import Image
from tqdm import tqdm

from src.dataset import SYSUDatasetGenerator, RegDBDatasetGenerator,\
    TestData, process_gallery_sysu, process_query_sysu, process_test_regdb,\
    genidx
from src.evalfunc import test 
from src.models.mvd import MVD
from src.models.trainingcell import CriterionWithNet, OptimizerWithNetAndCriterion
from src.loss import OriTripletLoss, CenterTripletLoss
from src.utils import IdentitySampler, genidx, AverageMeter, get_param_list,\
     LRScheduler


def get_parser():
    '''
    return a parser
    '''
    psr = argparse.ArgumentParser(description="DDAG Code Mindspore Version")

    # dataset settings
    psr.add_argument("--dataset", default='SYSU', choices=['SYSU', 'RegDB'],
                     help='dataset name: RegDB or SYSU')
    psr.add_argument('--data-path', type=str, default='data')
    # Only used on Huawei Cloud OBS service,
    # when this is set, --data_path is overridden by --data-url
    psr.add_argument("--data-url", type=str, default=None)
    psr.add_argument('--batch-size', default=4, type=int,
                     metavar='B', help='the number of person IDs in a batch')
    psr.add_argument('--test-batch', default=64, type=int,
                     metavar='tb', help='testing batch size')
    psr.add_argument('--num-pos', default=4, type=int,
                     help='num of pos per identity in each modality')
    psr.add_argument('--trial', default=1, type=int,
                     metavar='t', help='trial (only for RegDB dataset)')


    # image transform
    psr.add_argument('--img-w', default=144, type=int,
                     metavar='imgw', help='img width')
    psr.add_argument('--img-h', default=288, type=int,
                     metavar='imgh', help='img height')


    # model
    psr.add_argument('--arch', default='resnet50', type=str,
                     help='network baseline:resnet50')
    psr.add_argument('--z-dim', default=512, type=int,
                     help='information bottleneck z dim')


    # loss setting
    psr.add_argument('--loss-func', default="id+tri", type=str,
                     help='specify loss function type', choices=["id", "id+tri", "id+tri+kldiv"])
    psr.add_argument('--drop', default=0.2, type=float,
                     metavar='drop', help='dropout ratio')
    psr.add_argument('--margin', default=0.3, type=float,
                     metavar='margin', help='triplet loss margin')


    # optimizer and scheduler
    psr.add_argument("--lr", default=0.00035, type=float,
                     help='learning rate, 0.0035 for adam; 0.1 for sgd')
    psr.add_argument('--optim', default='adam', type=str, choices=['adam', 'sgd'],
                     help='optimizer')
    psr.add_argument("--warmup-steps", default=5, type=int,
                     help='warmup steps')
    psr.add_argument("--start-decay", default=15, type=int,
                     help='weight decay start epoch(included)')
    psr.add_argument("--end-decay", default=27, type=int,
                     help='weight decay end epoch(included)')
    psr.add_argument("--decay-factor", default=0.5,
                     help=r'lr_{epoch} = args.decay_factor * lr_{epoch - 1}')

    # training configs
    psr.add_argument('--epoch', default=80, type=int,
                     metavar='epoch', help='epoch num')
    psr.add_argument('--start-epoch', default=1, type=int,
                     help='start training epoch')
    psr.add_argument('--device-target', default="GPU",
                     choices=["GPU", "Ascend"])
    psr.add_argument('--gpu', default='0', type=str,
                     help='set CUDA_VISIBLE_DEVICES')

    # Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU).
    #  If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be
    # the number set in the environment variable 'CUDA_VISIBLE_DEVICES'.
    #  For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the 'device_id' can be 0,1,2 at the
    # moment, 'device_id' starts from 0, and 'device_id'=0 means using GPU of number 4.
    psr.add_argument('--device-id', default=0, type=str, help='')

    psr.add_argument('--device-num', default=1, type=int,
                     help='the total number of available gpus')
    psr.add_argument('--resume', '-r', default='', type=str,
                     help='resume from checkpoint, no resume:""')
    psr.add_argument('--pretrain', type=str,
                     default="",
                     help='Pretrain resnet-50 checkpoint path, no pretrain: ""')
    psr.add_argument('--run_distribute', action='store_true',
                     help="if true, will be run on distributed architecture with mindspore")
    psr.add_argument('--parameter-server', default=False)
    psr.add_argument('--save-period', default=20, type=int,
                     help=" save checkpoint file every args.save_period epochs")


    # logging configs
    psr.add_argument('--tag', default='toy', type=str, help='logfile suffix name')
    psr.add_argument("--branch-name", default="main",
                     help="Github branch name, for ablation study tagging")

    # testing / evaluation config
    psr.add_argument('--sysu_mode', default='all', type=str,
                     help=' test all or indoor search(only for SYSU-MM01)')
    psr.add_argument('--regdb_mode', default='v2i', type=str, choices=["v2i", "i2v"],\
        help='v2i: visible to infrared search; i2v:infrared to visible search.(Only for RegDB)')

    return psr


def print_dataset_info(dtype, trainset_, query_label_, gall_label_, start_time_):
    """
    This method print dataset information.
    """
    n_class_ = len(np.unique(trainset_.train_color_label))
    nquery_ = len(query_label_)
    ngall_ = len(gall_label_)
    print(f'Dataset {dtype} statistics:')
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print(f'  visible  | {n_class_:5d} | {len(trainset_.train_color_label):8d}')
    print(f'  thermal  | {n_class_:5d} | {len(trainset_.train_thermal_label):8d}')
    print('  ------------------------------')
    print(f'  query    | {len(np.unique(query_label_)):5d} | {nquery_:8d}')
    print(f'  gallery  | {len(np.unique(gall_label_)):5d} | {ngall_:8d}')
    print('  ------------------------------')
    print(f'Data Loading Time:\t {time.time() - start_time_:.3f}')


def decode(img):
    '''
    params:
        img: img of Tensor
    Returns:
        PIL image
    '''
    return Image.fromarray(img)


def optim(epoch_, backbone_lr_scheduler_, head_lr_scheduler_):
    '''
    return an optimizer of SGD or ADAM
    '''
    ########################################################################
    # Define optimizers
    ########################################################################
    epoch_ = ms.Tensor(epoch_, ms.int32)
    backbone_lr = float(backbone_lr_scheduler_(epoch_).asnumpy())
    head_lr = float(head_lr_scheduler_(epoch_).asnumpy())

    if args.optim == 'sgd':

        opt_p = SGD([
            {'params': net.rgb_backbone.trainable_params(), 'lr': backbone_lr},
            {'params': net.ir_backbone.trainable_params(), 'lr': backbone_lr},
            {'params': net.shared_backbone.trainable_params(), 'lr': backbone_lr},
            {'params': net.rgb_bottleneck.trainable_params(), 'lr': head_lr},
            {'params': net.ir_bottleneck.trainable_params(), 'lr': head_lr},
            {'params': net.shared_bottleneck.trainable_params(), 'lr': head_lr},
            ],
                    learning_rate=args.lr, weight_decay=5e-4, nesterov=True, momentum=0.9)

    elif args.optim == 'adam':

        opt_p = Adam([
            {'params': net.rgb_backbone.trainable_params(), 'lr': backbone_lr},
            {'params': net.ir_backbone.trainable_params(), 'lr': backbone_lr},
            {'params': net.shared_backbone.trainable_params(), 'lr': backbone_lr},
            {'params': net.rgb_bottleneck.trainable_params(), 'lr': head_lr},
            {'params': net.ir_bottleneck.trainable_params(), 'lr': head_lr},
            {'params': net.shared_bottleneck.trainable_params(), 'lr': head_lr},
        ],
            learning_rate=args.lr, weight_decay=5e-4)

    return opt_p


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.device_target == 'GPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ########################################################################
    # Init context
    ########################################################################
    device = args.device_target
    # init context
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=device, save_graphs=False)
    context.set_context(mode=context.GRAPH_MODE, device_target=device, save_graphs=False)

    if device == "CPU":
        LOCAL_DATA_PATH = args.data_path
        args.run_distribute = False
    else:
        if device == "GPU":
            LOCAL_DATA_PATH = args.data_path
            context.set_context(device_id=args.device_id)

        if args.parameter_server:
            context.set_ps_context(enable_ps=True)

        # distributed running context setting
        if args.run_distribute:
            # Ascend target
            if device == "Ascend":
                if args.device_num > 1:
                    # not useful now, because we only have one Ascend Device
                    pass
            # end of if args.device_num > 1:
                init()
            # GPU target
            else:
                init()
                context.set_auto_parallel_context(
                    device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                    gradients_mean=True
                )
                # mixed precision setting
                context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        # end of if target="Ascend":
    # end of if args.run_distribute:

        # Adapt to Huawei Cloud: download data from obs to local location
        if device == "Ascend":
            # Adapt to Cloud: used for downloading data from OBS to docker on the cloud
            import moxing as mox

            LOCAL_DATA_PATH = "/cache/data"
            args.data_path = LOCAL_DATA_PATH
            print("Download data...")
            mox.file.copy_parallel(src_url=args.data_url, dst_url=LOCAL_DATA_PATH)
            print("Download complete!(#^.^#)")
            # print(os.listdir(local_data_path))


    ########################################################################
    # Logging
    ########################################################################
    loader_batch = args.batch_size * args.num_pos

    if device in ("GPU", "CPU"):
        checkpoint_path = os.path.join("logs", args.tag, "training")
        os.makedirs(checkpoint_path, exist_ok=True)

        SUFFIX = str(args.dataset)

        SUFFIX = SUFFIX + f'_batch-size_2*{args.batch_size}*{args.num_pos}={2 * loader_batch}'
        SUFFIX = SUFFIX + f'_{args.optim}_lr_{args.lr}'
        SUFFIX = SUFFIX + f'_loss-func_{args.loss_func}'

        if args.dataset == 'RegDB':
            SUFFIX = SUFFIX + f'_trial_{args.trial}'

        SUFFIX = SUFFIX + "_" + args.branch_name

        time_msg = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_file = open(osp.join(checkpoint_path, f"{SUFFIX}_performance_{time_msg}.txt"),
                        "w", encoding="utf-8")
        print(f'Args: {args}')
        print(f'Args: {args}', file=log_file)


    ########################################################################
    # Create Dataset
    ########################################################################
    dataset_type = args.dataset

    if dataset_type == "SYSU":
        data_path = args.data_path
    elif dataset_type == "RegDB":
        data_path = args.data_path

    START_EPOCH = args.start_epoch
    start_time = time.time()

    print("==> Loading data")
    # Data Loading code

    transform_train_rgb = Compose(
        [
            decode,
            # py_trans.Pad(10),
            # py_trans.RandomCrop((args.img_h, args.img_w)),
            py_trans.RandomGrayscale(prob=0.5),
            py_trans.RandomHorizontalFlip(),
            py_trans.ToTensor(),
            py_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            py_trans.RandomErasing(prob=0.5)
        ]
    )

    transform_train_ir = Compose(
        [
            decode,
            # py_trans.Pad(10),
            # py_trans.RandomCrop((args.img_h, args.img_w)),
            # py_trans.RandomGrayscale(prob=0.5),
            py_trans.RandomHorizontalFlip(),
            py_trans.ToTensor(),
            py_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            py_trans.RandomErasing(prob=0.5)
        ]
    )


    transform_test = Compose(
        [
            decode,
            py_trans.Resize((args.img_h, args.img_w)),
            py_trans.ToTensor(),
            py_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    if dataset_type == "SYSU":
        # train_set
        trainset_generator = SYSUDatasetGenerator(data_dir=data_path)
        color_pos, thermal_pos = genidx(trainset_generator.train_color_label,\
            trainset_generator.train_thermal_label)

        # testing set
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.sysu_mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path,\
            mode=args.sysu_mode, random_seed=0)

    elif dataset_type == "RegDB":
        # train_set
        trainset_generator = RegDBDatasetGenerator(data_dir=data_path, trial=args.trial)
        color_pos, thermal_pos = genidx(trainset_generator.train_color_label,\
            trainset_generator.train_thermal_label)

        # testing set
        if args.regdb_mode == "v2i":
            query_img, query_label = process_test_regdb(img_dir=data_path,\
                modal="visible", trial=args.trial)
            gall_img, gall_label = process_test_regdb(img_dir=data_path,\
                modal="thermal", trial=args.trial)
        elif args.regdb_mode == "i2v":
            query_img, query_label = process_test_regdb(img_dir=data_path,\
                modal="thermal", trial=args.trial)
            gall_img, gall_label = process_test_regdb(img_dir=data_path,\
                modal="visible", trial=args.trial)

    ########################################################################
    # Create Query && Gallery
    ########################################################################

    gallset_generator = TestData(gall_img, gall_label, img_size=(args.img_w, args.img_h))
    queryset_generator = TestData(query_img, query_label, img_size=(args.img_w, args.img_h))

    print_dataset_info(dataset_type, trainset_generator, query_label, gall_label, start_time)

    ########################################################################
    # Define net
    ########################################################################

    # pretrain
    if args.pretrain != "":
        print(f"Pretrain model: {args.pretrain}")
        print(f"Pretrain model: {args.pretrain}", file=log_file)

    print('==> Building model..')
    n_class = len(np.unique(trainset_generator.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)

    net = MVD(num_class=n_class, drop=args.drop, z_dim=args.z_dim,\
         pretrain=args.pretrain)

    if args.resume != "":
        print(f"Resume checkpoint:{args.resume}")
        print(f"Resume checkpoint:{args.resume}", file=log_file)
        param_dict = load_checkpoint(args.resume)
        load_param_into_net(net, param_dict)
        if args.resume.split("/")[-1].split("_")[0] != "best":
            args.resume = int(args.resume.split("/")[-1].split("_")[1])
        print(f"Start epoch: {args.resume}")
        print(f"Start epoch: {args.resume}", file=log_file)


    ########################################################################
    # Define loss
    ########################################################################
    CELossNet = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    OriTripLossNet = OriTripletLoss(margin=args.margin, batch_size=2 * loader_batch)
    CenterTripLossNet = CenterTripletLoss(margin=args.margin, batch_size=2 * loader_batch)
    KL_Div = P.KLDivLoss()
    # TripLossNet = TripletLoss(margin=args.margin)

    net_with_criterion = CriterionWithNet(net, CELossNet,\
            OriTripLossNet, KL_Div, loss_func=args.loss_func)

    ########################################################################
    # Define schedulers
    ########################################################################

    assert (args.start_decay > args.warmup_steps) and (args.start_decay < args.end_decay) \
         and (args.end_decay < args.epoch)

    backbone_lr_scheduler = LRScheduler(args.lr, args.warmup_steps,\
        [args.start_decay, args.end_decay], args.decay_factor)
    head_lr_scheduler = LRScheduler(10 * args.lr, args.warmup_steps,\
        [args.start_decay, args.end_decay], args.decay_factor)

    ########################################################################
    # Start Training
    ########################################################################

    print('==> Start Training...')
    BEST_MAP = 0.0
    BEST_R1 = 0.0
    BEST_EPOCH = 0
    for epoch in range(START_EPOCH, args.epoch + 1):

        optimizer_P = optim(epoch, backbone_lr_scheduler, head_lr_scheduler)
        net_with_optim = OptimizerWithNetAndCriterion(net_with_criterion, optimizer_P)

        print('==> Preparing Data Loader...')
        # identity sampler:
        sampler = IdentitySampler(trainset_generator.train_color_label,\
            trainset_generator.train_thermal_label,\
            color_pos, thermal_pos, args.num_pos, args.batch_size)

        trainset_generator.cindex = sampler.index1  # color index
        trainset_generator.tindex = sampler.index2  # thermal index

        # add sampler
        trainset = ds.GeneratorDataset(trainset_generator,\
            ["color", "thermal", "color_label", "thermal_label"], sampler=sampler)

        trainset = trainset.map(operations=transform_train_rgb, input_columns=["color"])
        trainset = trainset.map(operations=transform_train_ir, input_columns=["thermal"])


        trainset.cIndex = sampler.index1  # color index
        trainset.tIndex = sampler.index2  # infrared index
        print(f"Epoch [{str(epoch)}]")

        # define callbacks
        loss_cb = LossMonitor()
        cb = [loss_cb]

        trainset = trainset.batch(batch_size=loader_batch, drop_remainder=True)

        dataset_helper = DatasetHelper(trainset, dataset_sink_mode=False)
        # net_with_optim = connect_network_with_dataset(net_with_optim, dataset_helper)

        net.set_train(mode=True)

        BATCH_IDX = 0
        N = np.maximum(len(trainset_generator.train_color_label),\
            len(trainset_generator.train_thermal_label))
        total_batch = int(N / loader_batch) + 1
        print("The total number of batch is ->", total_batch)

        # calculate average batch time
        batch_time = AverageMeter()
        end_time = time.time()

        # calculate average accuracy
        acc = AverageMeter()
        id_loss = AverageMeter()
        tri_loss = AverageMeter()


        ########################################################################
        # Batch Training
        ########################################################################
        time_msg = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        print('==>' + time_msg)
        print('==>' + time_msg, file=log_file)
        print('==> Start Training...')
        print('==> Start Training...', file=log_file)
        log_file.flush()
        
        BEST_MAP = 0.0
        BEST_R1 = 0.0
        BEST_EPOCH = 0
        best_param_list = None
        best_path = None
        
        for BATCH_IDX, (img1, img2, label1, label2) in enumerate(tqdm(dataset_helper)):
            label1 = ms.Tensor(label1, dtype=ms.float32)
            label2 = ms.Tensor(label2, dtype=ms.float32)
            img1, img2 = ms.Tensor(img1, dtype=ms.float32), ms.Tensor(img2, dtype=ms.float32)

            loss = net_with_optim(img1, img2, label1, label2)

            acc.update(net_with_criterion.acc)
            id_loss.update(net_with_criterion.loss_id.asnumpy())
            tri_loss.update(net_with_criterion.loss_tri.asnumpy())

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            if BATCH_IDX % 100 == 0:
                print('Epoch: [{}][{}/{}]   '
                      'Convolution LR: {CLR:.7f}   '
                      'IB & Classifier LR: {HLR:.7f}    '
                      'Loss:{Loss:.4f}   '
                      'id:{id:.4f}   '
                      'tri:{tri:.4f}   '
                      'Batch Time:{batch_time:.2f}  '
                      'Accuracy:{acc:.2f}   '
                      .format(epoch, BATCH_IDX, total_batch,\
                              CLR=float(backbone_lr_scheduler(ms.Tensor(epoch, ms.int32)).asnumpy()),
                              HLR=float(head_lr_scheduler(ms.Tensor(epoch, ms.int32)).asnumpy()),
                              Loss=float(loss.asnumpy()),
                              id=float(id_loss.avg),
                              tri=float(tri_loss.avg),
                              batch_time=batch_time.avg,
                              acc=float(acc.avg.asnumpy() * 100)
                            ))
                print('Epoch: [{}][{}/{}]   '
                      'Convolution LR: {CLR:.7f}   '
                      'IB & Classifier LR: {HLR:.7f}    '
                      'Loss:{Loss:.4f}   '
                      'id:{id:.4f}   '
                      'tri:{tri:.4f}   '
                      'Batch Time:{batch_time:.2f}  '
                      'Accuracy:{acc:.2f}   '
                      .format(epoch, BATCH_IDX, total_batch,\
                              CLR=float(backbone_lr_scheduler(ms.Tensor(epoch, ms.int32)).asnumpy()),
                              HLR=float(head_lr_scheduler(ms.Tensor(epoch, ms.int32)).asnumpy()),
                              Loss=float(loss.asnumpy()),
                              id=float(id_loss.avg),
                              tri=float(tri_loss.avg),
                              batch_time=batch_time.avg,
                              acc=float(acc.avg.asnumpy() * 100)
                            ))
                log_file.flush()

        ########################################################################
        # Epoch Evaluation
        ########################################################################

        if epoch > 0:

            net.set_train(mode=False)
            gallset = ds.GeneratorDataset(gallset_generator, ["img", "label"])
            gallset = gallset.map(operations=transform_test, input_columns=["img"])
            gallery_loader = gallset.batch(batch_size=args.test_batch)

            queryset = ds.GeneratorDataset(queryset_generator, ["img", "label"])
            queryset = queryset.map(operations=transform_test, input_columns=["img"])
            query_loader = queryset.batch(batch_size=args.test_batch)

            if args.dataset == "SYSU":
                cmc_v, mAP_v, cmc_i, mAP_i = test(args, gallery_loader, query_loader, ngall,\
                    nquery, net, gallery_cam=gall_cam, query_cam=query_cam)

            if args.dataset == "RegDB":
                cmc_v, mAP_v, cmc_i, mAP_i = test(args, gallery_loader, query_loader, ngall,\
                    nquery, net)

            print('v&v_ms:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}|\
                 Rank-20: {:.2%}| mAP: {:.2%}'.format(
                     cmc_v[0], cmc_v[4], cmc_v[9], cmc_v[19], mAP_v))
            print('v&v_ms:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}|\
                 Rank-20: {:.2%}| mAP: {:.2%}'.format(
                     cmc_v[0], cmc_v[4], cmc_v[9], cmc_v[19], mAP_v), file=log_file)


            print('i&i_ms:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}|\
                 Rank-20: {:.2%}| mAP: {:.2%}'.format(
                     cmc_i[0], cmc_i[4], cmc_i[9], cmc_i[19], mAP_i))
            print('i&i_ms:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}|\
                 Rank-20: {:.2%}| mAP: {:.2%}'.format(
                     cmc_i[0], cmc_i[4], cmc_i[9], cmc_i[19], mAP_i), file=log_file)


            mAP = (mAP_v + mAP_i) / 2.0
            cmc = (cmc_v + cmc_i) / 2.0

            print(f"rank-1: {cmc[0]:.2%}, mAP: {mAP:.2%}")
            print(f"rank-1: {cmc[0]:.2%}, mAP: {mAP:.2%}", file=log_file)

            # Save checkpoint weights every args.save_period Epoch
            save_param_list = get_param_list(net)
            if (epoch >= 2) and (epoch % args.save_period) == 0:
                path = osp.join(checkpoint_path,\
                    f"epoch_{epoch:02}_rank1_{cmc[0]*100:.2f}_mAP_{mAP*100:.2f}_{SUFFIX}.ckpt")
                save_checkpoint(save_param_list, path)

            # Record the best performance
            if mAP > BEST_MAP:
                BEST_MAP = mAP

            if cmc[0] > BEST_R1:
                best_param_list = save_param_list
                best_path = osp.join(checkpoint_path,\
                    f"best_epoch_{epoch:02}_rank1_{cmc[0]*100:.2f}_mAP_{mAP*100:.2f}_{SUFFIX}.ckpt")
                BEST_R1 = cmc[0]
                BEST_EPOCH = epoch


            print("******************************************************************************")
            print("******************************************************************************",
                  file=log_file)

            log_file.flush()


    print("=> Save best parameters...")
    print("=> Save best parameters...", file=log_file)
    save_checkpoint(best_param_list, best_path)
    
    if args.dataset == "SYSU":
        print(f"For SYSU-MM01 {args.sysu_mode} search, the testing result is:")
        print(f"For SYSU-MM01 {args.sysu_mode} search, the testing result is:", file=log_file)
    elif args.dataset == "RegDB":
        print(f"For RegDB {args.regdb_mode} search, the testing result is:")
        print(f"For RegDB {args.regdb_mode} search, the testing result is:", file=log_file)

    print(f"Best: rank-1: {BEST_R1:.2%}, mAP: {BEST_MAP:.2%}, \
        Best epoch: {BEST_EPOCH}(according to Rank-1)")
    print(f"Best: rank-1: {BEST_R1:.2%}, mAP: {BEST_MAP:.2%}, \
        Best epoch: {BEST_EPOCH}(according to Rank-1)", file=log_file)
    log_file.flush()
    log_file.close()
