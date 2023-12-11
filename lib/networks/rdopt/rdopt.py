import time
import os
import random as rd
from lib.config import cfg
import neural_renderer as nr
import numpy as np
import torch
from torch import nn
import cv2
from .backbone import Backbone
# from .repose_feat import ReposeFeat
from .gn import GNLayer
# from lib.networks.rdopt.texture_net import TextureNet
from lib.csrc.camera_jacobian.camera_jacobian_gpu import calculate_jacobian
from .util import rot_vec_to_mat, spatial_gradient, crop_input, crop_features
from lib.datasets.dataset_catalog import DatasetCatalog
import pycocotools.coco as coco
from lib.utils.pvnet import pvnet_config
from lib.utils import img_utils
from lib.utils.pvnet import pvnet_pose_utils
from lib.networks.rdopt import unet
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import time
from torch.nn import functional as nnF
#from .losses import scaled_barron
import torchvision
import math

# import tensorrt as trt
# logger = trt.Logger(trt.Logger.INFO)
# from torch2trt import TRTModule
# import torch
# import numpy as np
#lossfn = scaled_barron(0, 0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mean = pvnet_config.mean
std = pvnet_config.std
rd.seed(5215)
viridis = cm.get_cmap('viridis', 256)

GPU_COUNT = torch.cuda.device_count()
MAX_NUM_OF_GN = 5
IN_CHANNELS = 3
OUT_CHANNELS = 6
VER_NUM = 5841  # 35034#
scalefactor = 0.15  # 这个应该由感受野最大响应范围决定 默认0.17


class RDOPT(nn.Module):
    def __init__(self):
        super(RDOPT, self).__init__()

        filename = f'./data/linemod/{cfg.model}/{cfg.model}.obj'
        vertices, faces, textures = nr.load_obj(filename)

        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])
        self.register_buffer('textures', textures[None, :, :])

        # self.renderer=[]
        self.scales = [1, 4]
        self.train_scales = [1, 4]
        # self.K=[]
        # for k in range(len(self.scales)):
        #         self.K.append( torch.tensor([[572.4114/self.scales[k],   0.0000,  (325.2611+0.5)*self.scales[k]-0.5],
        # [  0.0000, 573.5704/self.scales[k],(242.0490+0.5)*self.scales[k]-0.5 ],
        # [  0.0000,   0.0000,   1.0000]]) )
        #         self.renderer.append(nr.Renderer(image_height=int(cfg.image_height/self.scales[k]),
        #                             image_width=int(cfg.image_width/self.scales[k]),
        #                             camera_mode='projection'))
        # self.renderer = nr.Renderer(image_height=NEWSHAPE,
        #                             image_width=NEWSHAPE,
        #                             camera_mode='projection')

        self.rendererori = nr.Renderer(image_height=256,
                                       image_width=256,
                                       camera_mode='projection')
        # self.texture_net = TextureNet(IN_CHANNELS, OUT_CHANNELS, vertices)
        # self.repose_feat = ReposeFeat(OUT_CHANNELS)
        self.gn = GNLayer(OUT_CHANNELS)

        split = 'test'
        if split == 'test':
            args = DatasetCatalog.get(cfg.test.dataset)
        else:
            args = DatasetCatalog.get(cfg.train.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

        confupdated = {'name': 'unet', 'trainable': True, 'freeze_batch_normalization': False,
                       'output_scales': [0, 2, 4], 'output_dim': [32, 128, 128], 'encoder': 'vgg16',
                       'num_downsample': 4, 'decoder': [64, 64, 64, 32], 'decoder_norm': 'nn.BatchNorm2d',
                       'do_average_pooling': False, 'compute_uncertainty': True, 'checkpointed': False}
        conf2 = {'name': 'unet', 'encoder': 'vgg16', 'decoder': [64, 64, 64, 32], 'output_scales': [0, 2],
                 'output_dim': [OUT_CHANNELS, OUT_CHANNELS, OUT_CHANNELS], 'freeze_batch_normalization': False,
                 'do_average_pooling': False, 'compute_uncertainty': True, 'checkpointed': True}

        self.UNet_ = unet.UNet(conf2)

        self.normalize_features = True
        self.original_textures = textures

        # f = open("refinednew.trt", "rb")
        # runtime = trt.Runtime(logger)
        # engine = runtime.deserialize_cuda_engine(f.read())
        # self.trt_model = TRTModule(engine, input_names=['input'], output_names=['output'])
        #
        # uncertainty = []
        # block_rough = AdaptationBlock(OUT_CHANNELS, 1)
        # block_refine = AdaptationBlock(OUT_CHANNELS, 1)
        # uncertainty.append(block_rough)
        # uncertainty.append(block_refine)
        #
        # self.uncertainty = nn.ModuleList(uncertainty)
        #  if conf.compute_uncertainty:
        #   self.uncertainty = nn.ModuleList(uncertainty)

        # self.scales = [2 ** s for s in conf2['output_scales']]

        fcrough = nn.Linear(in_features=6 * 6, out_features=6, bias=False)
        fcfine = nn.Linear(in_features=6 * 6, out_features=6, bias=False)
        fc = []
        fc.append(fcrough)
        fc.append(fcfine)
        self.fc = nn.ModuleList(fc)
        self.miu = []

    def updatemiu(self, miu, c, cons):

        midval = miu * (c ** 2)
        M = ((2 * midval) / (2 + midval)) - cons

        deltamiu = (midval * M + 2 * M - 2 * midval) / ((c ** 2) * M - 2 * (c ** 2))

        finalmiu = -deltamiu + miu
        # print('miuuuuu',finalmiu)
        # if finalmiu<1:
        #     finalmiu=1

        return torch.relu(finalmiu - 1) + 1

    def corr_matching(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()

        # reshape features for matrix multiplication
        feature_A = feature_A.view(b, c, h * w).transpose(1, 2)  # size [b,c,h*w]
        feature_B = feature_B.view(b, c, h * w)  # size [b,c,h*w]
        # perform matrix mult.
        feature_mul = torch.bmm(feature_A, feature_B)
        # indexed [batch,row_A,col_A,row_B,col_B]
        correlation_tensor = feature_mul.view(b, h, w, h, w).unsqueeze(1)

        if True:
            correlation_tensor = nnF.normalize(nn.ReLU()(correlation_tensor), dim=1)

        return correlation_tensor

    def dense_feature_extraction(self, image, name, scale_image):

        # image = image.astype(np.float32)  # better for resizing
        scale_resize = (1., 1.)
        #  self.conf.resize=256
        if False:  # 这也许是处理图像不是一个正方形的情况，但是原代码是true，但没有进入下面，因为 (max(image.shape[:2]) > target_size or
            #         self.conf.resize_by == 'max_force')这里判断是false
            pass
            # target_size = self.conf.resize // scale_image
            # if (max(image.shape[:2]) > target_size or
            #         self.conf.resize_by == 'max_force'):
            #     image, scale_resize = resize(image, target_size, max, 'linear')

        import time
        #  time_start = time.time()

        pred = self.UNet_(image)  # features, scales, weight

        features = pred['feature_maps']
        assert len(self.scales) == len(features)

        features = [feat.squeeze(0) for feat in features]  # remove batch dim
        confidences = pred.get('confidences')  # [-2.5, 2]

        if confidences is not None:
            confidences = [c.squeeze(0) for c in confidences]

        scales = [(scale_resize[0] / s, scale_resize[1] / s)
                  for s in self.scales]
        weight = confidences

        #    time_end = time.time()
        # print('time cost2', time_end - time_start, 's')
        # self.log_dense(name=name, image=image, image_scale=image_scale,
        #    features=features, scales=scales, weight=weight)

        if True:
            assert weight is not None
            # stack them into a single tensor (makes the bookkeeping easier)
            features = [torch.cat([f, w], 0) for f, w in zip(features, weight)]

        # Filter out some layers or keep them all
        # if self.conf.layer_indices is not None:
        #     features = [features[i] for i in self.conf.layer_indices]
        #     scales = [scales[i] for i in self.conf.layer_indices]

        return features, scales

    def visualize(self, R_, t_, step, batchid, batchinp):
        inp = img_utils.unnormalize_img(batchinp, mean,
                                        std).permute(1, 2, 0)
        # mask = output['mask'][0].detach().cpu().numpy()

        R = rot_vec_to_mat(R_).detach().cpu().numpy()
        t = t_.detach().cpu().numpy()[:, :, None]
        pose_preds = np.concatenate((R, t), axis=2)

        img_id = batchid
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        pose_gt = np.array(anno['pose'])
        K = np.array(anno['K'])

        cond1 = np.linalg.norm(pose_preds[-1, :, 3] - pose_preds[0, :, 3]) > 0.03
        cond2 = np.linalg.norm(pose_preds[-1, :, 3] - pose_gt[:, 3]) < 0.01

        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        _, ax = plt.subplots(1)
        ax.imshow(inp)

        for i, pose_pred in enumerate(pose_preds):
            color = viridis((i + 1) / 7)
            corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
            ax.add_patch(
                patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]],
                                fill=False,
                                linewidth=1,
                                edgecolor=color))
            ax.add_patch(
                patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]],
                                fill=False,
                                linewidth=1,
                                edgecolor=color))

        os.makedirs(f'linemod/samples/{cfg.test.dataset}/{cfg.cls_type}', exist_ok=True)

        ax.add_patch(
            patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]],
                            fill=False,
                            linewidth=1,
                            edgecolor='g'))
        ax.add_patch(
            patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]],
                            fill=False,
                            linewidth=1,
                            edgecolor='g'))
        plt.savefig(f'linemod/samples/{cfg.test.dataset}/{cfg.cls_type}/{cfg.cls_type}_{img_id}_{step}.png')

    def rend_feature(self, vertices, faces, textures, K, R_vec, t, bbox, num=0):
        R_mat = rot_vec_to_mat(R_vec)
        t = t.view(-1, 1, 3)
        #   print(vertices.shape, faces.shape, textures.shape, K.shape, R_mat.shape, t.shape, bbox[:, 0:1].shape, bbox[:, 1:2].shape)

        f_rend, face_index_map, depth_map = self.renderer(
            vertices, faces, textures, K, R_mat, t, bbox[:, 0:1], bbox[:, 1:2])
        mask = f_rend[:, -1:]
        f_rend = f_rend[:, :-1]

        return f_rend, mask, R_mat, face_index_map, depth_map

    def rend_featureori(self, vertices, faces, textures, K, R_vec, t, bbox, num=0):
        R_mat = rot_vec_to_mat(R_vec)
        t = t.view(-1, 1, 3)
        #   print(vertices.shape, faces.shape, textures.shape, K.shape, R_mat.shape, t.shape, bbox[:, 0:1].shape, bbox[:, 1:2].shape)

        f_rend, face_index_map, depth_map = self.rendererori(
            vertices, faces, textures, K, R_mat, t, bbox[:, 0:1], bbox[:, 1:2])
        mask = f_rend[:, -1:]
        f_rend = f_rend[:, :-1]

        return f_rend, mask, R_mat, face_index_map, depth_map

    def show_features(self, features, name='', num=0):  # input is ([1, 32, 256, 256])

        tmpf = features[num].transpose(0, 2)[:, :, 0:3]
        tmpf = np.array(tmpf.cpu().detach())
        tmpffinal = (tmpf - tmpf.min()) / (tmpf.max() - tmpf.min())
        cv2.imwrite('./f_rend' + name + '.png', tmpffinal * 255)

    def show_features_pure(self, features, name='', num=0):  # input is ([1, 32, 256, 256])

        tmpf = features[num].transpose(0, 2)[:, :, 0:3]
        tmpf = np.array(tmpf.cpu().detach())
        cv2.imwrite('./f_rend' + name + '.png', tmpf * 255)

    def show_index_map(self, index_map, name=''):  # input is torch.Size([16, 256, 256])
        if index_map.shape[0] > 2:
            tmpf = index_map.transpose(0, 2)[:, :, 0:3]
            tmpf = np.array(tmpf.cpu().detach())
            tmpffinal = (tmpf - tmpf.min()) / (tmpf.max() - tmpf.min())
            cv2.imwrite('./f_index_map' + name + '.png', tmpffinal * 255)
        else:
            print('wrong')

    # def show_mask(self, tmpf,name=''):# input is torch.Size([256, 256])
    #         tmpf = tmpf.transpose(0, 1)
    #         tmpf=np.array(tmpf.cpu().detach())
    #         tmpffinal = (tmpf - tmpf.min()) / (tmpf.max() - tmpf.min())
    #         cv2.imwrite('./f_mask'+name+'.png', tmpffinal * 255)
    def show_mask2(self, tmpf, name=''):  # input is torch.Size([1, 256, 256])
        tmpf = tmpf.transpose(0, 2)
        tmpf = np.array(tmpf.cpu().detach())
        tmpffinal = (tmpf - tmpf.min()) / (tmpf.max() - tmpf.min())
        cv2.imwrite('./f_mask' + name + '.png', tmpffinal * 255)

    # 得搞清楚repose 和pixeloc 各钱箱传播几次  pixeloc 一次大循环中unet的forward只传播了两次（query一次，ref一次）；而repose理论上需要N+1次，1次是针对ref的unet的forward，N次是query的unet的forward,N is iteration number

    def loss_calculation(self, R_gt, t_gt, R, t, vertices=None, faces=None, textures=None, K=None, bbox=None,
                         matchloss_=None):

        Rm_gt = rot_vec_to_mat(R_gt).transpose(2, 1)
        v_gt = torch.add(torch.bmm(vertices.expand(K.shape[0], VER_NUM, 3), Rm_gt), t_gt.view(-1, 1, 3))
        ################################################################################################################################
        # f_rend, r_mask, R_mat, face_index_map, depth_map = \
        #     self.rend_feature(vertices.expand(K.shape[0], VER_NUM, 3), faces.expand(K.shape[0], 11678, 3),
        #                       textures.expand(K.shape[0], VER_NUM, 3),
        #                       K, R_gt, t_gt, bbox)
        #   self.show_features(f_rend,'fir')
        ############################################################################################################

        # R_ini = x[:, :3]
        # t_ini =  x[:, 3:].view(-1, 1, 3)
        # Rm_ini = rot_vec_to_mat(R_ini).transpose(2, 1)
        # v_ini = torch.add(torch.bmm(vertices.expand(K.shape[0], VER_NUM, 3), Rm_ini), t_ini)

        Rm = rot_vec_to_mat(R).transpose(2, 1)
        v = torch.add(torch.bmm(vertices.expand(K.shape[0], VER_NUM, 3), Rm), t.view(-1, 1, 3))
        ############################################################################################################
        # f_rend, r_mask, R_mat, face_index_map, depth_map = \
        #     self.rend_feature(vertices.expand(K.shape[0], VER_NUM, 3), faces.expand(K.shape[0], 11678, 3),
        #                       textures.expand(K.shape[0], VER_NUM, 3),
        #                       K, R, t, bbox)
        #   self.show_features(f_rend,'sec')
        ############################################################################################################

        # v_ini = v_ini.view(-1, 3)
        # v_gt = v_gt.view(-1, 3)

        if cfg.cls_type not in ['eggbox', 'glue']:
            # pose_ini_loss = torch.norm(v_ini - v_gt, 2, -1).mean(dim=1)
            pose_loss = torch.norm(v - v_gt, 2, -1).mean(dim=1)
        else:
            # pose_ini_cdist = torch.cdist(v_ini, v_gt, 2)
            pose_cdist = torch.cdist(v, v_gt, 2)
            # pose_ini_loss = torch.min(pose_ini_cdist, dim=1)[0].mean(dim=1)
            pose_loss = torch.min(pose_cdist, dim=1)[0].mean(dim=1)

        # scalar_stats.update({'pose_ini_loss': pose_ini_loss.mean()})
        #    scalar_stats.update({'pose_loss': pose_loss.mean()})

        if matchloss_ != None:

            return 10 * pose_loss + matchloss_

        else:
            return pose_loss

    def rouconvert(self, c, miu, z):
        up = miu * (c ** 2)
        downA = miu * (c ** 2) + z
        sqrtrou = up / downA
        # if sqrtrou.any()<0 or miu<0:
        #     print('wrong')

        return sqrtrou ** 2

    def forward(self, inp_ori, K, x_ini, oribbox, x2s, x4s, x8s, xfc, R_gt=None, t_gt=None, batchid=None, maskori=None,
                epoch=None):
        torch.backends.cudnn.benchmark = True  # 这个竟然还真的有点用
        time_startall = time.time()
        # newbbox=torch.tensor([ [  -96, -96, 96, 96]]).cuda()
        newlocation = torch.tensor([[0.0, 0.0, 1.3]]).cuda()
        fx = K[0][0][0]
        fy = K[0][1][1]
        cx = K[0][0][2]
        cy = K[0][1][2]
        newK = torch.tensor([[[fx, 0.0000, 0], [0.0000, fy, 0], [0.0, 0.0, 1.0]]]).cuda()

        def process_siamese(data_i, featuresrequired=False):
            # time_sub1 = time.time()

            # inputdata = {}
            # inputdata['image'] = data_i

            pred_i, features = self.UNet_(data_i)  # features, scales, weight
            #    pred_i = self.trt_model(data_i)
            # transintooxnn=True
            # if transintooxnn:
            #
            #
            #     input_name = ['input']
            #     output_name = ['output']
            #     input = torch.randn(1, 3, 256, 256).cuda()
            #     model = self.UNet_.eval()
            #     torch.onnx.export(model, input, "/home/yuan/doc/objectpose/meshpose/RePOSE/finedpose.onnx",  opset_version=10,input_names=input_name, output_names=output_name,export_params=True,
            #                        )
            if featuresrequired:
                return pred_i, features
            else:
                return pred_i

        output = {}
        matchloss_ = None

        bs, _, h, w = inp_ori.shape

        vertices = self.vertices
        faces = self.faces
        # textures = self.textures
        textures = self.original_textures.unsqueeze(0)  # self.texture_net()# texture forwarded

        # inp = crop_input(inp_ori*mask.unsqueeze( dim=0).cuda(), oribbox)

        # cv2.imwrite('./f_mask_input.png',np.array(mask.squeeze()*255)  )

        ###################################################################################这边不能用膨胀，而应该用开运算mask
        kernel = np.ones((3, 3), np.uint8)
        dict = cv2.dilate(maskori.cpu().transpose(0, 2).numpy(), kernel, iterations=20)
        if bs == 1:
            dict = dict[:, :, None]
        mask = torch.tensor(dict).transpose(0, 2).cuda()

        # tmpffinal = cv2.resize(tmpffinal, (0, 0), fx=1.2, fy=1.2, interpolation=cv2.INTER_NEAREST)
        ###################################################################################这边不能用膨胀，而应该用开运算mask
        mask = crop_input(mask.unsqueeze(dim=0).transpose(0, 1), oribbox)
        # maskori = crop_input(maskori.unsqueeze(dim=0).transpose(0,1), bbox)
        # dd=maskori.cpu()[:,0,:,:].transpose(0, 2).numpy()
        # w0, h0, _ = dd.shape
        # dict_ = cv2.resize(dd, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
        # w, h, _ = dict_.shape
        # # print('dffffffffff',(w-w0)/2,(w-(w-w0)/2),(h-h0)/2,(h-(h-h0)/2))
        # dict_ = dict_[int((w - w0) / 2):int(w - (w - w0) / 2), int((h - h0) / 2):int(h - (h - h0) / 2), :]
        # mask = torch.tensor(dict_).transpose(0, 2).unsqueeze(0).transpose(0, 1).cuda()

        inpsub_ori = crop_input(inp_ori, oribbox)
        #  cv2.imwrite('./f_inp_input.png',  ( np.transpose(np.array(inp.cpu())[0], (2, 1, 0)) - np.transpose(np.array(inp.cpu())[0],  (2, 1, 0)).min()) * 255 / ( np.transpose(np.array(inp.cpu())[0], (2, 1, 0)).max() - np.transpose(np.array(inp.cpu())[0], (2, 1, 0)).min()))

        x = torch.zeros_like(x_ini)
        x[:, :3] = x_ini[:, :3]
        x[:, 3:] = x_ini[:, 3:]
        ##################################################################
        x[:, :3] = R_gt
        x[:, 3:] = t_gt.view(-1, 1, 3).squeeze()

        #
        # x[:, 0 ]= x[:, 0]+0.3*rd.random()-0.15
        # x[:, 1 ]= x[:, 1]+0.3*rd.random()-0.15
        # x[:, 2 ]= x[:, 2]+0.3*rd.random()-0.15
        #
        # x[:, -1] = x[:, -1]+0.03*rd.random()-0.015
        # x[:, -2] = x[:, -2]+0.03*rd.random()-0.015
        # x[:, -3] = x[:, -3]+0.03*rd.random()-0.015

        for kk in range(bs):
            x[kk, 0] = x[kk, 0] + 1 * rd.random() - 0.5
            x[kk, 1] = x[kk, 1] + 1 * rd.random() - 0.5
            x[kk, 2] = x[kk, 2] + 1 * rd.random() - 0.5

            x[kk, -1] = x[kk, -1] + 0.1 * rd.random() - 0.05
            x[kk, -2] = x[kk, -2] + 0.1 * rd.random() - 0.05
            x[kk, -3] = x[kk, -3] + 0.1 * rd.random() - 0.05

            # x[kk, 0 ] = x[kk, 0]+0.5*rd.random()-0.25
            # x[kk, 1 ] = x[kk, 1]+0.5*rd.random()-0.25
            # x[kk, 2 ] = x[kk, 2]+0.5*rd.random()-0.25
            #
            # x[kk, -1] = x[kk, -1]+0.03*rd.random()-0.015
            # x[kk, -2] = x[kk, -2]+0.03*rd.random()-0.015
            # x[kk, -3] = x[kk, -3]+0.03*rd.random()-0.015

        # x[:, 0 ]= x[:, 0]+1*rd.random()-0.5
        # x[:, 1 ]= x[:, 1]+1*rd.random()-0.5
        # x[:, 2 ]= x[:, 2]+1*rd.random()-0.5
        #
        # x[:, -1] = x[:, -1]+0.1*rd.random()-0.05
        # x[:, -2] = x[:, -2]+0.1*rd.random()-0.05
        # x[:, -3] = x[:, -3]+0.1*rd.random()-0.05
        xinit = x
        if self.training:
            with torch.no_grad():  # 这个后面应该要去掉，耗时！！！
                #  print('xinitxinitxinit',xinit,R_gt, t_gt.squeeze())
                pose_loss_init = self.loss_calculation(R_gt, t_gt.squeeze(), xinit[:, :3], xinit[:, 3:], vertices,
                                                       faces,
                                                       textures, K, oribbox)  #
        ##################################################################
        if x[0, 5] < 0.5:  # ?????? 啥意思？
            print('?????????????')
            output['R'] = x[:, :3]
            output['t'] = x[:, 3:]
            output['vertices'] = vertices
            return output, 0.0, False

        if not self.training:
            x_all = torch.zeros((MAX_NUM_OF_GN * 2 + 1, 6), device=x.device)
            x_all[0] = x[0]
            xallcount = 0
        #    self.visualize(x[:, :3], x[:, 3:], xallcount, batchid, inp_ori[0])

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # time_start = time.time()

        # features_query, scales_query = self.dense_feature_extraction(
        #     torch.tensor(np.ones([1,3,256,256])).cuda().float(), 'test.png', 1)#  images_ref[idx].shape  (768, 1024, 3)
        #
        # print('55555555',features_query[0].max(),features_query[0].min())
        inputdata = []
        inputdata.append(inpsub_ori)
        pred = {}

        # inputdata.append(f_rend)
        AA, reffeaturesori = process_siamese(inputdata[0], featuresrequired=True)
        pred_ori = {'ref': AA}  # , 'query': process_siamese(inputdata[1])
        # pred = {}
        #
        # # inputdata.append(f_rend)
        # pred = {'ref': process_siamese(inputdata[0])}  # , 'query': process_siamese(inputdata[1])
        # #  self.show_features(pred['query']['feature_maps'][0])  #self.show_features(pred['query']['confidences'][0])
        #
        # # self.show_index_map(mask[1] * pred['ref']['feature_maps'][0][1])
        # x_scale_output = []
        # renderfeatures = []
        # renderconfidences = []
        # verticesmid = []
        losses = 0.0
        steplength = 0

        # self.miu = 8000000*torch.ones(bs,1,256,256).cuda()
        for i in reversed(range(len(self.scales))):
            # if i==2  :
            #     continue
            time_endsub1 = time.time()
            # F_ref_ori = pred_ori['ref']['feature_maps'][i]
            # W_ref_ori = pred_ori['ref']['confidences'][i]
            for j in range(MAX_NUM_OF_GN):
                scaled_value = int(-x[:, 3:][0][2] / scalefactor) + 12
                NEWSHAPE = 16 * scaled_value  # 16 times

                with torch.no_grad():
                    # newu1 = cx + (-64*newlocation[0][2] + fx * x[:, 3:][0][0]) / x[:, 3:][0][2]  # (-64,-64) 对应的u
                    # newv1 = cy + (-64*newlocation[0][2] + fx * x[:, 3:][0][1]) / x[:, 3:][0][2]  # (-64,-64) 对应的v
                    #
                    # newu2 = cx + (64*newlocation[0][2] + fx * x[:, 3:][0][0]) / x[:, 3:][0][2]  # ( 64, 64) 对应的u
                    # newv2 = cy + (64*newlocation[0][2] + fx * x[:, 3:][0][1]) / x[:, 3:][0][2]  # ( 64, 64) 对应的v
                    newu1 = cx + (-(NEWSHAPE / 2) * x[:, 3:][0][2] + fx * x[:, 3:][0][0]) / x[:, 3:][0][
                        2]  # (-64,-64) 对应的u
                    newv1 = cy + (-(NEWSHAPE / 2) * x[:, 3:][0][2] + fy * x[:, 3:][0][1]) / x[:, 3:][0][
                        2]  # (-64,-64) 对应的v

                    newu2 = cx + ((NEWSHAPE / 2) * x[:, 3:][0][2] + fx * x[:, 3:][0][0]) / x[:, 3:][0][
                        2]  # ( 64, 64) 对应的u
                    newv2 = cy + ((NEWSHAPE / 2) * x[:, 3:][0][2] + fy * x[:, 3:][0][1]) / x[:, 3:][0][
                        2]  # ( 64, 64) 对应的v

                    bbox = torch.tensor(
                        [[int(newv1 + 0.5), int(newu1 + 0.5), int(newv2 + 0.5), int(newu2 + 0.5)]]).cuda()
                    # inp = torchvision.transforms.functional.crop(inp_ori,
                    #                                                 int(newv1 + 0.5)  ,
                    #                                                 int(newu1 + 0.5)  , int(newv2 - newv1 + 0.5),
                    #                                                 int(newu2 - newu1 + 0.5))
                    # inp = crop_input(inp_ori, bbox)
                    #  print('ssssssss',inp.shape,x,newu1,newu2,newv1,newv2)
                    #  inputdata = []
                    #  inputdata.append(inp)
                    #  pred = {}
                    #
                    #
                    #  pred = {'ref': process_siamese(inputdata[0])}  # , 'query': process_siamese(inputdata[1])
                    #  F_ref = pred['ref']['feature_maps'][i]
                    #  W_ref = pred['ref']['confidences'][i]
                    ##################################################注意！！！！u,v这里要掉个位置，因为输入图片inp_ori和bbox就已经转zhi了,
                    #   print('rrrrrrrrrrr', int(newv1 + 0.5) - bbox[0][0],int(newu1 + 0.5) - bbox[0][1])
                if int(newv1 + 0.5) - oribbox[0][0] < -128 or int(newv1 + 0.5) - oribbox[0][0] > 256 or int(
                        newv2 - newv1 + 0.5) > 256:
                    break

                F_ref_1 = torchvision.transforms.functional.crop(pred_ori['ref']['feature_maps'][i],
                                                                 int(newv1 + 0.5) - oribbox[0][0],
                                                                 int(newu1 + 0.5) - oribbox[0][1],
                                                                 int(newv2 - newv1 + 0.5),
                                                                 int(newu2 - newu1 + 0.5))
                W_ref_1 = torchvision.transforms.functional.crop(pred_ori['ref']['confidences'][i],
                                                                 int(newv1 + 0.5) - oribbox[0][0],
                                                                 int(newu1 + 0.5) - oribbox[0][1],
                                                                 int(newv2 - newv1 + 0.5),
                                                                 int(newu2 - newu1 + 0.5))
                F_ref = pred_ori['ref']['feature_maps'][i]
                W_ref = pred_ori['ref']['confidences'][i]
                # F_ref = nnF.interpolate(F_ref, size=(192, 192), mode='bilinear', align_corners=True)
                # W_ref = nnF.interpolate(W_ref, size=(192, 192), mode='bilinear', align_corners=True)
                # W_ref = nnF.interpolate(W_ref, size=(192, 192), mode='bilinear', align_corners=True)
                #
                if self.normalize_features:
                    F_ref_1 = nnF.normalize(F_ref_1,
                                            dim=1)  # B x C x W x H   将输入的数据（input）按照指定的维度(dim)做p范数(默认是2范数)运算，即将某一个维度除以那个维度对应的范数。
                if self.normalize_features:
                    F_ref = nnF.normalize(F_ref,
                                          dim=1)  # B x C x W x H   将输入的数据（input）按照指定的维度(dim)做p范数(默认是2范数)运算，即将某一个维度除以那个维度对应的范数。

                #  time_sub1 = time.time()
                with torch.no_grad():

                    # f_rend, r_mask, R_mat, face_index_map, depth_map = \
                    #     self.rend_feature(vertices.expand(K.shape[0], VER_NUM, 3),
                    #                          faces.expand(K.shape[0], 11678, 3),
                    #                          textures.expand(K.shape[0], VER_NUM, 3),
                    #                          K, x[:, :3], x[:, 3:], bbox)

                    f_rendori, r_mask, R_mat, face_index_mapori, depth_mapori = \
                        self.rend_featureori(vertices.expand(K.shape[0], VER_NUM, 3),
                                             faces.expand(K.shape[0], 11678, 3),
                                             textures.expand(K.shape[0], VER_NUM, 3),
                                             K, x[:, :3], x[:, 3:], oribbox)  # 这个好像是对的

                    f_rend = torchvision.transforms.functional.crop(f_rendori,
                                                                    int(newv1 + 0.5) - oribbox[0][0],
                                                                    int(newu1 + 0.5) - oribbox[0][1],
                                                                    int(newv2 - newv1 + 0.5),
                                                                    int(newu2 - newu1 + 0.5))  # 这个好像是对的

                    face_index_map = torchvision.transforms.functional.crop(face_index_mapori,
                                                                            int(newv1 + 0.5) - oribbox[0][0],
                                                                            int(newu1 + 0.5) - oribbox[0][1],
                                                                            int(newv2 - newv1 + 0.5),
                                                                            int(newu2 - newu1 + 0.5))  # 这个好像是对的
                    depth_map = torchvision.transforms.functional.crop(depth_mapori,
                                                                       int(newv1 + 0.5) - oribbox[0][0],
                                                                       int(newu1 + 0.5) - oribbox[0][1],
                                                                       int(newv2 - newv1 + 0.5),
                                                                       int(newu2 - newu1 + 0.5))  # 这个好像是对的

                    # cropped_bbox = torch.tensor([[int(newv1 + 0.5), int(newu1 + 0.5), int(newv2 + 0.5) , int(newu2 + 0.5) ]]).cuda()
                    # centerv= ( newv1+newv2)/2
                    # centeru= ( newu1+newu2)/2
                    # deltav=int( 0.5+centerv-(bbox[0][2]+bbox[0][0])/2)
                    # deltau=int( 0.5+centeru-(bbox[0][3]+bbox[0][1])/2)
                    # print('sssssss',  int(newv1 + 0.5) - bbox[0][0],
                    #                                        int(newu1 + 0.5) - bbox[0][1], int(newv2 - newv1 + 0.5),
                    #                                        int(newu2 - newu1 + 0.5))
                    # print('sssssss',  int(newv1 + 0.5) - bbox[0][0],
                    #                                        int(newu1 + 0.5) - bbox[0][1], int(newv2 - newv1 + 0.5),
                    #                                        int(newu2 - newu1 + 0.5))
                    # f_rend, r_mask, R_mat, face_index_map, depth_map = \
                    #     self.rend_feature(vertices.expand(K.shape[0], VER_NUM, 3), faces.expand(K.shape[0], 11678, 3),
                    #                       textures.expand(K.shape[0], VER_NUM, 3),
                    #                       newK, x[:, :3] , torch.tensor([ [ 0.0, 0.0, x[:, 3:][0][2]]]).cuda(), newbbox)#不能render到中心，这是错的，因为由于zhui的视角，看的物体的部位可能不一样
                # maskori.cpu()[:,0,:,:].transpose(0, 2).numpy()
                #    time_sub2 = time.time()
                # print('time sub', f_rend.shape,x[:, 3:][0], (-128*x[:, 3:][0][2] + fy * x[:, 3:][0][1]),x[:, 3:][0][2] ,cy,cx)
                #   print('sssss',f_rend.shape,int(newv1 + 0.5) - oribbox[0][0],
                #                                                 int(newu1 + 0.5) - oribbox[0][1],
                #                                                int(newv2 - newv1 + 0.5),
                #                                               int(newu2 - newu1 + 0.5),x)
                queryresult, queryrfeatures = process_siamese(f_rend, featuresrequired=True)  # 0.0023
                queryresultori = process_siamese(f_rendori)  # 0.0023
                ####################################
                # import torchvision
                # queryresult = process_siamese(torchvision.transforms.functional.crop(f_rend, 80, 60, 128, 128))
                # self.UNet_.half(torchvision.transforms.functional.crop(f_rend, 80, 60, 128, 128))
                # queryresult=pred['ref']
                ####################################

                F_q = queryresultori['feature_maps'][i]

                if self.normalize_features:  # TRue  但是opt里面的确实false
                    F_q = nnF.normalize(F_q, dim=1)  # B x C x W x H

                W_q = queryresultori['confidences'][i]

                reffeaturesori_sub = nnF.interpolate(reffeaturesori[-1], size=(256, 256), mode='bilinear',
                                                     align_corners=True)
                reffeaturesori_sub = torchvision.transforms.functional.crop(reffeaturesori_sub,
                                                                            int(newv1 + 0.5) -
                                                                            oribbox[0][0],
                                                                            int(newu1 + 0.5) -
                                                                            oribbox[0][1],
                                                                            int(newv2 - newv1 + 0.5),
                                                                            int(newu2 - newu1 + 0.5))
                reffeaturesori_sub = nnF.interpolate(reffeaturesori_sub,
                                                     size=(queryrfeatures[-1].shape[-1], queryrfeatures[-1].shape[-1]),
                                                     mode='bilinear', align_corners=True)

                # rentaestimation=torch.cdist(queryrfeatures, referencefeatures, p=2)
                sumresult = nnF.normalize(queryrfeatures[-1], dim=1) * nnF.normalize(reffeaturesori_sub, dim=1)

                sumresult = sumresult.sum(1)
                sumresult = nnF.interpolate(sumresult.unsqueeze(1), size=(6, 6), mode='bilinear', align_corners=True)
                renta = self.fc[i](sumresult.squeeze(1).flatten(1))

                min_ = -6
                max_ = 5
                lambda_ = 10. ** (min_ + renta.sigmoid() * (max_ - min_))

                # print(referencefeatures.shape)
                # self.show_features(F_ref,'F_ref'+str(batchid)+str(i))
                # self.show_features(F_q,'F_q'+str(batchid)+str(i))

                # matchresult=self.corr_matching(  F_ref, F_q)
                # W_q = torchvision.transforms.functional.affine(W_q, translate=(deltau, deltav), angle=0, scale=1,  shear=0)
                # F_q = torchvision.transforms.functional.affine(F_q, translate=(deltau, deltav), angle=0, scale=1,shear=0)
                #
                # W_ref = torchvision.transforms.functional.affine(W_ref, translate=(deltau, deltav), angle=0, scale=1,
                #                                                       shear=0)
                # F_ref = torchvision.transforms.functional.affine(F_ref,  translate=(deltau, deltav), angle=0, scale=1,
                #                                                       shear=0)
                #  W_ref = W_ref * mask.float()
                #

                c_rou_d = W_ref * W_q  # confidences   感觉可能不用xW_q，因为CNN对query tmp边上的梯度都已经是0了，而且tmp应该全是正确的，也就是没有噪声;confidence是mask作用的，而不是提取特征作用的

                min_ = -2
                max_ = 2
                c_rou_d = 10. ** (min_ + c_rou_d * (max_ - min_))
                z = (F_q - F_ref) ** 2

                with torch.no_grad():
                    self.miu = 398 / (c_rou_d ** 2)
                    self.miu = torch.relu(self.miu - 1) + 1
                    steplength =  0.4 * (j+ 1)#((epoch - 1) / 120) * 2  # 大循环的思路中steplength 设为0~2，随着迭代步长而变化，然后一直为2
                   # print('dddddddddddddd',steplength,epoch)
                    self.miu = self.updatemiu(self.miu, c_rou_d, steplength)

                if torch.isinf(self.miu).any() == True:
                    print('wrong')
                #  print('sssssssssssss',self.miu.max(),self.miu.min())
                weight = self.rouconvert(c_rou_d, self.miu, z.sum(1).unsqueeze(1))

                e = (F_ref - F_q)
                # cost, w_loss, _ = lossfn((e**2).sum(1))
                # weight=weight*w_loss.unsqueeze(0)
                # print('losssssssssssssssssss',diff_loss.sum(dim=1).sum())

                grad_xy = spatial_gradient(-F_q)
                # Perform anlytical jacobian computation
                feature_dim = grad_xy.shape[-2]
                # J_c = calculate_jacobian(face_index_map, depth_map, K, R_mat,
                #                          x[:, :3].contiguous(), x[:, 3:].contiguous(), cropped_bbox.int())
                # J_c = calculate_jacobian(face_index_map, depth_map,  K, R_mat,
                #                          x[:, :3].contiguous(),x[:, 3:].contiguous(), cropped_bbox.int())
                J_c = calculate_jacobian(face_index_mapori.contiguous(), depth_mapori.contiguous(), K, R_mat,
                                         x[:, :3].contiguous(), x[:, 3:].contiguous(), oribbox)
                e = e.permute((0, 2, 3, 1))
                weight = weight.permute((0, 2, 3, 1))
                e = e.reshape(bs, -1, feature_dim)
                weight = weight.reshape(bs, -1, 1)
                grad_xy = grad_xy.view(-1, feature_dim, 2)
                J_c = J_c.view(-1, 2, 6)
                J = torch.bmm(grad_xy,
                              J_c)  # torch.Size([32768, 16, 2]);torch.Size([1048576, 2, 6])    torch.Size([1048576, 3, 2]) torch.Size([1048576, 2, 6])
                J = J.reshape(bs, -1, OUT_CHANNELS, 6)

                x_update = self.gn(x, e, J, weight, i, lambda_)  # 0.03458
                x = x_update
                #    self.miu=self.updatemiu(self.miu,c_rou_d,0.5 )#logmiu,c,constant

                if not self.training:
                    x_all[xallcount + 1] = x[0]
                    xallcount = xallcount + 1

                #  self.visualize(x[:, :3], x[:, 3:], xallcount, batchid, inp_ori[0])
            #  x_scale_output.append(x_update)
            # renderfeatures.append(F_q )
            # renderconfidences.append(W_q)
            # verticesmid.append(vertices.expand(K.shape[0], VER_NUM, 3))
            if self.training:
                #   print('xnext', i, x,R_gt, t_gt.squeeze())

                # e = (F_ref - F_q)  # e = (F_ref - F_q)*weight
                # matchloss = W_ref * W_q  * e
                # matchloss = matchloss ** 2#torch.Size([2, 6, 256, 256])
                # matchloss = matchloss.contiguous().view(bs, -1)
                # matchloss = matchloss.sum(dim=1)
                # mask_ = mask.view(bs, -1)
                # mask_sum = mask_.sum(dim=1)
                # matchloss = matchloss / (mask_sum + 1e-10)
                # matchloss_ = matchloss.mean()

                pose_loss_ori = self.loss_calculation(R_gt, t_gt.squeeze(), x[:, :3], x[:, 3:], vertices, faces,
                                                      textures, K, oribbox, matchloss_)  #

                pose_loss = pose_loss_ori / len(self.scales)

                if i < len(self.scales) - 1:
                    pose_loss = pose_loss * success.float()
                self.success_thresh = 0.0075
                thresh = self.success_thresh * self.train_scales[i]  # 这个是保证每次迭代精度是越来越高的
                # print('fffffffffffff', self.extractor.scales[-1 - i], -1 - i)  # 16/4/1        -1,-2,-3
                success = pose_loss_ori < thresh  # 这样做是保证，第N次迭代的loss是否考虑进去取决于第N-1次迭代是否成功，如果第N-1次没成功，则总的loss就不会将第N次的loss考虑进去，也就是不去学第N次的loss,也就是不学第N次的迭代（这个和什么都不做的全loss的学习优shi在于，每次迭代肯定是在有着较好的初值情况下（因为是确保了第N-1次成功的）来学习的，而不是随机初值）

                print('loss', i, pose_loss_init, pose_loss_ori, thresh, success)
                losses += pose_loss

            x = x_update.detach()  # ?????c

            time_endsub2 = time.time()

        # print('time costsub', time_endsub2 - time_endsub1, 's')
        if self.training:
            losses = torch.mean(losses) * 100
        #   print('gggggggg', losses)

        if not self.training:
            output['R'] = x[:, :3]
            output['t'] = x[:, 3:]
            output['R_all'] = x_all[:, :3]
            output['t_all'] = x_all[:, 3:]

        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        print( 'length and epoch',steplength, epoch)
        return losses, output, elapsed_time  # x_scale_output,renderfeatures ,renderconfidences,verticesmid #, elapsed_time, True


def get_res_rdopt():
    model = RDOPT()
    return model
