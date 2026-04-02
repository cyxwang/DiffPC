import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

from os.path import join as fullfile


try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


class CMPAllDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LR_paths,self.SF_paths0,self.SF_paths125,self.SF_paths126, self.GT_paths = [], [], [] ,[],[]
        self.LR_env,self.SF_env, self.GT_env = None, None, None  # environment for lmdb
        self.LR_size, self.SF_size, self.GT_size = opt["LR_size"],opt["SF_size"], opt["GT_size"]
     
        self.dataname =  opt["dataname"]
        self.dataroot = opt['dataroot']
        # self.data_num = opt['datanum']
        self.datasize = 256 # opt['data_size']
  
        self.dataset_len = opt["len"] # *len(self.dataname)
        self.subfold = opt['subfold']
        self.len = 0

        # read image list from lmdb or image files
        if 'train' in opt['name']:
            self.split = 'train'
        else:
            self.split = 'test'
        for j, datapath in enumerate(self.dataroot):
            for i, name in enumerate(self.dataname[j]):
                if 'warp' in self.subfold[j] :
                    hr_path = util.get_image_paths(opt["data_type"],fullfile(datapath,self.split))
                    lr_path = util.get_image_paths(opt["data_type"],fullfile(datapath,name,'cam',self.subfold[j] ,self.split))
                    sr_path = util.get_image_paths(opt["data_type"],fullfile(datapath,name,'cam',self.subfold[j] ,'ref'))
                else:
                    lr_path = util.get_image_paths(opt["data_type"],fullfile(datapath,'train'))
                    hr_path = util.get_image_paths(opt["data_type"],fullfile(datapath,name,'prj/cmp_last_iter',))
                    sr_path = util.get_image_paths(opt["data_type"],fullfile(datapath,name,'cam/warpSL/','ref'))
                surface0 = sr_path[0]
                surface125 = sr_path[-2]
                surface126 = sr_path[-1]
                sr_path0,sr_path125,sr_path126 = [],[],[]

                if self.dataset_len >0:
                    hr_path = hr_path[:self.dataset_len]
                    lr_path = lr_path[:self.dataset_len]

                for sr in range (len(hr_path)):
                    sr_path0.append(surface0)
                    sr_path125.append(surface125)
                    sr_path126.append(surface126)

                self.GT_paths.extend(hr_path)
                self.LR_paths.extend(lr_path)
                self.SF_paths0.extend(sr_path0)
                self.SF_paths125.extend(sr_path125)
                self.SF_paths126.extend(sr_path126)



        self.len = len(self.GT_paths)
 
        self.random_scale_list = [1]


    def __getitem__(self, index):

        GT_path,SF_path0,SF_path125,SF_path126, LR_path = None, None, None, None, None

        GT_path = self.GT_paths[index]
        
        img_GT = util.read_img(GT_path,size=self.datasize)  # return: Numpy float32, HWC, BGR, [0,1]

        LR_path = self.LR_paths[index]
        
        img_LR = util.read_img( LR_path,size=self.datasize)
        
        SF_path0 = self.SF_paths0[index]

        SF_path125 = self.SF_paths125[index]
        
        img_SF_Black = util.read_img( SF_path0,size=self.datasize)
        
        img_SF_White = util.read_img( SF_path125,size=self.datasize)
        SF_path126 = self.SF_paths126[index]
        img_SF_Gray = util.read_img( SF_path126,size=self.datasize)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
            img_SF_Black = img_SF_Black[:, :, [2, 1, 0]]
            img_SF_White = img_SF_White[:, :, [2, 1, 0]]
            img_SF_Gray = img_SF_Gray[:, :, [2, 1, 0]]
            
        
        if self.split == 'train':
            [img_GT,img_LR,img_SF_Black,img_SF_White,img_SF_Gray]= util.augment_imlist([img_GT,img_LR,img_SF_Black,img_SF_White,img_SF_Gray],split = 'train')

        
        
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()
        img_SF_Black = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_SF_Black, (2, 0, 1)))
        ).float()
        img_SF_White = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_SF_White, (2, 0, 1)))
        ).float()
        img_SF_Gray = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_SF_Gray, (2, 0, 1)))
        ).float()

     
        return {"LQ": img_LR, "GT": img_GT,"SF_B": img_SF_Black,"SF_W": img_SF_White,"SF_G": img_SF_Gray, "LQ_path": LR_path, "GT_path": GT_path,"SF_B_path": SF_path0,"SF_W_path": SF_path125,"SF_G_path": SF_path126}

    def __len__(self):
        return len(self.GT_paths)
