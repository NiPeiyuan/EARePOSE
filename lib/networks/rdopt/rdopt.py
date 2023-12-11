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
# from .losses import scaled_barron
import torchvision
#import open3d
from lib.evaluators.linemod.rdopt import get_projection_2d,get_3d
# import tensorrt as trt
# logger = trt.Logger(trt.Logger.INFO)
# from torch2trt import TRTModule
# import torch
# import numpy as np
# lossfn = scaled_barron(0, 0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mean = pvnet_config.mean
std = pvnet_config.std
rd.seed(5215)
viridis = cm.get_cmap('viridis', 256)

GPU_COUNT = torch.cuda.device_count()
VER_NUM_dict ={'ape':5841,'cam':4750,'benchvise':4790,'can':2852,'driller':3165,'cat':15736,'eggbox':4619,'duck':7912,'iron':2277,'holepuncher':27305,'glue': 3740,'iron': 2277 ,'phone': 2071 ,'lamp': 3156  }
scalefactor_dict={'ape':0.15,'cam':0.08,'benchvise':0.08,'can':0.08,'driller':0.08,'cat':0.08 ,'eggbox': 0.08,'duck':0.08 ,'iron':0.08,'holepuncher':0.08,'glue': 0.08 ,'iron': 0.08  ,'phone': 0.08 ,'lamp': 0.08}
initialbias_dict={'ape':12,'cam':21,'benchvise':23,'can':22,'driller':23,'cat':20,'eggbox':19,'duck':19,'iron':22,'holepuncher':21,'glue':22,'iron': 22,'phone': 23,'lamp': 23}
face_num_dict={'ape':11678,'cam':9496,'benchvise':9580,'can':5708,'driller':6326,'cat':31468,'eggbox':9234,'duck':15820,'iron':4554,'holepuncher':467541,'glue':7476,'iron': 4554,'phone':4138,'lamp': 6124}


MAX_NUM_OF_GN = 5
IN_CHANNELS = 3
OUT_CHANNELS = 6
VER_NUM = VER_NUM_dict[cfg.cls_type]
scalefactor =scalefactor_dict[cfg.cls_type]
initialbias=initialbias_dict[cfg.cls_type]
face_num=face_num_dict[cfg.cls_type]
savednumpy=[]
savedindex=[]
import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
from torch2trt import TRTModule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

f = open('./bestresult/'+cfg.cls_type+'/'+cfg.cls_type+'.trt', "rb")
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(f.read())  
trt_model = TRTModule(engine, input_names=['input'], output_names=['output1', 'output2', 'output3', 'output4', 'output5'])

class RDOPT(nn.Module):
    def __init__(self):
        super(RDOPT, self).__init__()

        filename = f'./data/linemod/{cfg.model}/{cfg.model}.obj'
        vertices, faces, textures = nr.load_obj(filename)

        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])
        self.register_buffer('textures', textures[None, :, :])
 
        self.scales = [1, 4]
        self.train_scales = [1, 4]
 

        self.rendererori = nr.Renderer(image_height=256,
                                       image_width=256,
                                       camera_mode='projection')
    
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

        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=1))[0]
        self.corner_3d = np.array(anno['corner_3d'])
        self.center_3d = (np.max(self.corner_3d, 0) + np.min(self.corner_3d, 0)) / 2
        self.errorR = []
        self.errort = []
        self.maxR = []
        self.maxt = []
 

        fcrough_mid = nn.Linear(in_features=6 * 6 + 1, out_features=9, bias=False)
        fcfine_mid = nn.Linear(in_features=6 * 6 + 1, out_features=9, bias=False)
        fc_mid = []
        fc_mid.append(fcrough_mid)
        fc_mid.append(fcfine_mid)
        self.fc_mid = nn.ModuleList(fc_mid)

        fcrough = nn.Linear(in_features=6 * 6, out_features=6, bias=False)
        fcfine = nn.Linear(in_features=6 * 6, out_features=6, bias=False)
        fc = []
        fc.append(fcrough)
        fc.append(fcfine)
        self.fc = nn.ModuleList(fc)
        self.miu = []
        self.count = 0

 
 

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

   



    def process_siamese(self,data_i, featuresrequired=False):
 
            pred_1, pred_2, pred_3, pred_4 , features = trt_model(data_i)
            
            pred_i = {'feature_maps': [pred_1,pred_2]}
            pred_i['confidences'] = [pred_3,pred_4]

            if featuresrequired:
                return pred_i, features
            else:
                return pred_i




 
    def rouconvert(self, c, miu, z):
        up = miu * (c ** 2)
        down = miu * (c ** 2) + z
        sqrtrou = up / down 
        # if sqrtrou.any()<0 or miu<0:
        #     print('wrong') 
        return sqrtrou ** 2



    def refineonce(self,i,j,x,scalefactor,vertices,faces,textures,K,fx,fy,cx,cy,pred_ori,oribbox,reffeaturesori,bs,staterequired=False):

        scaled_value = int(-x[:, 3:][0][2] / scalefactor) + initialbias
        NEWSHAPE = 16 * scaled_value  

        scaled_valueorisize = -x[:, 3:][0][2] / scalefactor + initialbias
        NEWSHAPEorisize = 16 * scaled_valueorisize 
 

        with torch.no_grad():
        
            newu1 = cx + (-(NEWSHAPE / 2) * x[:, 3:][0][2] + fx * x[:, 3:][0][0]) / x[:, 3:][0][
                2]  # (-64,-64)  
            newv1 = cy + (-(NEWSHAPE / 2) * x[:, 3:][0][2] + fy * x[:, 3:][0][1]) / x[:, 3:][0][
                2]  # (-64,-64)  

            newu2 = cx + ((NEWSHAPE / 2) * x[:, 3:][0][2] + fx * x[:, 3:][0][0]) / x[:, 3:][0][
                2]  # ( 64, 64)  
            newv2 = cy + ((NEWSHAPE / 2) * x[:, 3:][0][2] + fy * x[:, 3:][0][1]) / x[:, 3:][0][
                2]  # ( 64, 64)  

            bbox = torch.tensor(
                [[int(newv1 + 0.5), int(newu1 + 0.5), int(newv2 + 0.5), int(newu2 + 0.5)]]).cuda()
 

            newu1orisize = cx + (-(NEWSHAPEorisize / 2) * x[:, 3:][0][2] + fx * x[:, 3:][0][0]) / x[:, 3:][0][
                2]  # (-64,-64)  
            newv1orisize = cy + (-(NEWSHAPEorisize / 2) * x[:, 3:][0][2] + fy * x[:, 3:][0][1]) / x[:, 3:][0][
                2]  # (-64,-64) 

            newu2orisize = cx + ((NEWSHAPEorisize / 2) * x[:, 3:][0][2] + fx * x[:, 3:][0][0]) / x[:, 3:][0][
                2]  # ( 64, 64)  
            newv2orisize = cy + ((NEWSHAPEorisize / 2) * x[:, 3:][0][2] + fy * x[:, 3:][0][1]) / x[:, 3:][0][
                2]  # ( 64, 64)  
            maxshapesize = 256 / 6
            minshapesize = 6 / 6
            scaleinput = torch.tensor((int(newv2orisize - newv1orisize + 0.5) / 6 - minshapesize) / (
                    maxshapesize - minshapesize)).cuda()
 

        F_ref = torchvision.transforms.functional.crop(pred_ori['ref']['feature_maps'][i],
                                                       int(newv1 + 0.5) - oribbox[0][0],
                                                       int(newu1 + 0.5) - oribbox[0][1],
                                                       int(newv2 - newv1 + 0.5),
                                                       int(newu2 - newu1 + 0.5))
        W_ref = torchvision.transforms.functional.crop(pred_ori['ref']['confidences'][i],
                                                       int(newv1 + 0.5) - oribbox[0][0],
                                                       int(newu1 + 0.5) - oribbox[0][1],
                                                       int(newv2 - newv1 + 0.5),
                                                       int(newu2 - newu1 + 0.5))

 
        if self.normalize_features:
            F_ref = nnF.normalize(F_ref,
                                  dim=1)  # B x C x W x H   

   
        with torch.no_grad():
 
            f_rend, r_mask, R_mat, face_index_map, depth_map = \
                self.rend_featureori(vertices.expand(K.shape[0], VER_NUM, 3),
                                     faces.expand(K.shape[0], face_num, 3),
                                     textures.expand(K.shape[0], VER_NUM, 3),
                                     K, x[:, :3], x[:, 3:], oribbox)  
 
            f_rend = torchvision.transforms.functional.crop(f_rend,
                                                            int(newv1 + 0.5) - oribbox[0][0],
                                                            int(newu1 + 0.5) - oribbox[0][1],
                                                            int(newv2 - newv1 + 0.5),
                                                            int(newu2 - newu1 + 0.5))  
            if int(newv1 + 0.5) - oribbox[0][0] < -NEWSHAPE or int(newv1 + 0.5) - oribbox[0][0] > 256 or int(
                    newu1 + 0.5) - oribbox[0][1] > 256 or int(newu1 + 0.5) - oribbox[0][1] < -NEWSHAPE:
                print('not converge')
                output['R'] = x[:, :3]
                output['t'] = x[:, 3:]
                output['vertices'] = vertices
                return 0.0, output, 0.0

 

            paddingnum = max(-(int(newv1 + 0.5) - oribbox[0][0]), -(int(newu1 + 0.5) - oribbox[0][1]),
                             int(newv1 + 0.5) - oribbox[0][0] + int(newv2 - newv1 + 0.5) - 256,
                             int(newu1 + 0.5) - oribbox[0][1] + int(newu2 - newu1 + 0.5) - 256)
            paddingnum = paddingnum.cpu().numpy().tolist()
            if paddingnum > 0:
                face_index_map = nnF.pad(face_index_map.float().unsqueeze(0),
                                         (paddingnum, paddingnum, paddingnum, paddingnum), "constant",
                                         -1).int().squeeze(1)

                depth_map = nnF.pad(depth_map.unsqueeze(0), (paddingnum, paddingnum, paddingnum, paddingnum),
                                    "constant", 100.0).squeeze(1)

            else:
 
                paddingnum = 0
 
            face_index_map = torchvision.transforms.functional.crop(face_index_map,
                                                                    paddingnum + int(newv1 + 0.5) - oribbox[0][
                                                                        0],
                                                                    paddingnum + int(newu1 + 0.5) - oribbox[0][
                                                                        1],
                                                                    int(newv2 - newv1 + 0.5),
                                                                    int(newu2 - newu1 + 0.5))  
 
            depth_map = torchvision.transforms.functional.crop(depth_map,
                                                               paddingnum + int(newv1 + 0.5) - oribbox[0][0],
                                                               paddingnum + int(newu1 + 0.5) - oribbox[0][1],
                                                               int(newv2 - newv1 + 0.5),
                                                               int(newu2 - newu1 + 0.5))   

        queryresult, queryrfeatures = self.process_siamese(f_rend, featuresrequired=True)   
 

        F_q = queryresult['feature_maps'][i]

        if self.normalize_features:  
            F_q = nnF.normalize(F_q, dim=1)  # B x C x W x H
 
        W_q = queryresult['confidences'][i]

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
 
        sumresult = nnF.normalize(queryrfeatures[-1], dim=1) * nnF.normalize(reffeaturesori_sub, dim=1)

        sumresult = sumresult.sum(1)
        sumresult = nnF.interpolate(sumresult.unsqueeze(1), size=(6, 6), mode='bilinear', align_corners=True)
        renta = self.fc[i](sumresult.squeeze(1).flatten(1))

        min_ = -6
        max_ = 5
        lambda_ = 10. ** (min_ + renta.sigmoid() * (max_ - min_))

        #weight= W_ref * W_q
        c_predicted = W_ref * W_q


        e = (F_ref - F_q)
        z = e ** 2
        weight = self.rouconvert(c_predicted, 1.0, z.sum(1).unsqueeze(1))#prediction mode# This is sqrt of rou function

        e = (F_ref - F_q)
 

        grad_xy = spatial_gradient(-F_q)
        # Perform anlytical jacobian computation
        feature_dim = grad_xy.shape[-2]

        J_c = calculate_jacobian(face_index_map.contiguous(), depth_map.contiguous(), K, R_mat,
                                 x[:, :3].contiguous(), x[:, 3:].contiguous(), bbox.int())

        e = e.permute((0, 2, 3, 1))
        weight = weight.permute((0, 2, 3, 1))
        e = e.reshape(bs, -1, feature_dim)
        weight = weight.reshape(bs, -1, 1)
        grad_xy = grad_xy.view(-1, feature_dim, 2)

        J_c = J_c.view(-1, 2, 6)
        J = torch.bmm(grad_xy,
                      J_c)  # torch.Size([32768, 16, 2]);torch.Size([1048576, 2, 6])    torch.Size([1048576, 3, 2]) torch.Size([1048576, 2, 6])
        J = J.reshape(bs, -1, OUT_CHANNELS, 6)

        x_update,delta_x = self.gn(x, e, J, weight, i, lambda_)  # 0.03458
        x = x_update


        if staterequired==True:#(10*delta_x[0]).cpu().numpy().tolist()         renta.sigmoid().cpu().numpy().tolist()  sumresult.squeeze(1).flatten(1)[0].tolist()  i

                # del_f=10 * delta_x[0]
                 defnew=torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0])
                 del_f=delta_x[0]
                 defnew[0:3] = del_f[0:3]*10
                 defnew[3::] = del_f[3::]*100

                 returnfeatures=defnew.sigmoid().cpu().numpy().tolist()

                 rentaf= renta.sigmoid().cpu().numpy().tolist()[0]

                 returnfeatures.extend(rentaf)

                 sumresultf=sumresult.squeeze(1).flatten(1)[0].tolist()
                 returnfeatures.extend(sumresultf)
                 returnfeatures.append(float(i))

                 return x,returnfeatures
        else:
                 return x



    def epsilon_greedy(self,state, q_network, epsilon=0.05):
        """Perform epsilon-greedy action sampling.

        Args:
            state: numpy ndarray, current state
            q_network: torch module

        Returns:
            action: one action
        """
        #   Epsilon-greedy action sampling

        r = np.random.random()
        if (r <= epsilon):
            action = np.random.choice(3)
        else:
            action = q_network.forward(state).argmax()
            action = action.item()
        return action







    def stepaction(self,action,x,batch,evaluator,diamter,totalitersteps,totalroughstep,totalrefinestep,is_terminal,buffer,returnstates,scalefactor, vertices, faces,
                                                  textures, K, fx, fy, cx, cy, pred_ori, oribbox,
                                                  reffeaturesori, bs):
        if action == 2:# stop
 
            is_terminal = True
            next_state = returnstates  
 

        if action == 1:  # coarse
            with torch.inference_mode():
                previousstates = returnstates

                x, returnstates = self.refineonce(1, totalroughstep, x, scalefactor, vertices, faces,
                                                  textures, K, fx, fy, cx, cy, pred_ori, oribbox,
                                                  reffeaturesori, bs, True)  # i,j 1,0
                totalroughstep = totalroughstep + 1
                totalitersteps = totalitersteps + 1

 
                if totalitersteps == 10:  
 
                   is_terminal = True
                   next_state = previousstates  #
        

                else:
    
                    is_terminal = False
                    next_state = returnstates  #
    

        if action == 0:  # fine
            with torch.inference_mode():
                previousstates = returnstates

                x, returnstates = self.refineonce(0, totalrefinestep, x, scalefactor, vertices, faces,
                                                  textures, K, fx, fy, cx, cy, pred_ori, oribbox,
                                                  reffeaturesori, bs, True)  # i,j 1,0
                totalrefinestep = totalrefinestep + 1
                totalitersteps = totalitersteps + 1

 
                if totalitersteps == 10:  
 
                    is_terminal = True
                    next_state = previousstates  #
 

                else:
     
                    is_terminal = False
                    next_state = returnstates  #
   

        return  is_terminal,buffer,x, returnstates,0,totalroughstep,totalrefinestep,totalitersteps,0








    def forward(self, inp_ori, K, x_ini, oribbox,  R_gt=None, t_gt=None, batchid=None, maskori=None, buffer=None,evaluator=None,batch=None,randomactions=False,q_network=None ,epi=None,epsilon=None):#buffer,evaluator,batch,randomactions
        torch.backends.cudnn.benchmark = True  
        time_startall = time.time()
        # newbbox=torch.tensor([ [  -96, -96, 96, 96]]).cuda()
        newlocation = torch.tensor([[0.0, 0.0, 1.3]]).cuda()
        fx = K[0][0][0]
        fy = K[0][1][1]
        cx = K[0][0][2]
        cy = K[0][1][2]
        newK = torch.tensor([[[fx, 0.0000, 0], [0.0000, fy, 0], [0.0, 0.0, 1.0]]]).cuda()







        output = {}
        matchloss_ = None

        bs, _, h, w = inp_ori.shape

        vertices = self.vertices
        faces = self.faces 
        textures = self.original_textures.unsqueeze(0)  # self.texture_net()# texture forwarded
 

        inpsub_ori = crop_input(inp_ori, oribbox)
 
        x = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).cuda()
        x[:, :3] = x_ini[:, :3] 
        x[:, 3:] = x_ini[:, 3:]   


        xinit = x

        ##################################################################

        if x[0, 5] < 0.5:  
 

            output['R'] = x[:, :3]
            output['t'] = x[:, 3:]
            output['vertices'] = vertices
            return  0.0,output, 0.0


        if not self.training:
            x_all = torch.zeros((MAX_NUM_OF_GN * 2 + 1, 6), device=x.device)
            x_all[0] = x[0]
            xallcount = 0


        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
       
        inputdata = []
        inputdata.append(inpsub_ori)
        pred = {}

 
        pred_ref_tmp, reffeaturesori = self.process_siamese(inpsub_ori, featuresrequired=True)


        pred_ori = {'ref': pred_ref_tmp}  # , 'query': process_siamese(inputdata[1])

 
        losses = 0.0
         #########################################################################################################################  env.reset()
        totalroughstep=0
        totalrefinestep=0
        totalitersteps=0
        with torch.inference_mode():

            x,returnstates = self.refineonce(1, totalroughstep, x, scalefactor, vertices, faces, textures, K, fx, fy, cx, cy, pred_ori, oribbox,
                                reffeaturesori, bs,True)
            totalitersteps =totalitersteps+1
            totalroughstep=totalroughstep+1
        if not self.training:
            # print(x)
            x_all[xallcount + 1] = x[0]
            xallcount = xallcount + 1
 
        diamter=evaluator.return_diamter(0.1)
       # stepcount = 0
        if randomactions:
              pass
        else:
 
            # Hyperparameters
             # epsilon = 0.05  # For epsilon start greedy action sampling
              epsilon_min = 0.05
              epsilon_decay = 0.99



              is_terminal=False
              rewards = []
              actions = []
  
              epsilon=0.0
              while not is_terminal:
                  action = self.epsilon_greedy(returnstates, q_network, epsilon=epsilon)
                  is_terminal, buffer, x, returnstates, reward, totalroughstep, totalrefinestep, totalitersteps,successornot = self.stepaction(
                      action, x, batch, evaluator, diamter, totalitersteps, totalroughstep, totalrefinestep,
                      is_terminal, buffer, returnstates, scalefactor, vertices, faces,
                      textures, K, fx, fy, cx, cy, pred_ori, oribbox,
                      reffeaturesori, bs)



                  rewards.append(reward)
                  actions.append(action)
 

              output['R'] = x[:, :3]
              output['t'] = x[:, 3:]



              end.record()
              torch.cuda.synchronize()
              elapsed_time = start.elapsed_time(end)
              #print('elapsed_time',elapsed_time)

              return returnstates,output, elapsed_time, epsilon,q_network, buffer,sum(rewards),actions,successornot
 

def get_res_rdopt():
    model = RDOPT()
    return model
