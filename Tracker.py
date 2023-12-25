import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer


class Tracker(object):
    """
    Tracker类为跟踪模块，共有四个主要函数：__init__()、optimize_cam_in_batch()、update_para_from_mapping()、run()函数;
    """
    def __init__(self, cfg, args, slam):
        # 初始化函数；
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.idx = slam.idx
        self.nice = slam.nice
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.seperate_LR = cfg['tracking']['seperate_LR']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    # 相机姿态的迭代优化，通过采样像素get_samples(),渲染深度，颜色，不确定性(render_batch_ray),计算loss并进行反向传播来完成；
    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        # 相机内外参初始化；
        device = self.device
        # 相机内参；
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        # 梯度归零；
        optimizer.zero_grad()
        # 相机外参；
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
        # 如果是NICE-SLAM,那么会过滤掉不在预定边界内的depth值；
        if self.nice:
            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

        # 调用render_batch_ray()函数进行渲染，得到预测的深度，不确定值，颜色；
        ret = self.renderer.render_batch_ray(
            self.c, self.decoders, batch_rays_d, batch_rays_o,  self.device, stage='color',  gt_depth=batch_gt_depth)
        depth, uncertainty, color = ret

        uncertainty = uncertainty.detach()
        # 如果使用了动态处理(self.handle_dynamic)，那么应用中值滤波方法来识别和处理动态对象；(来自于论文中一个消融实验）
        if self.handle_dynamic:
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        else:
            mask = batch_gt_depth > 0

        # 论文中的第一个损失函数：预测深度和真实深度之间的差值；
        loss = (torch.abs(batch_gt_depth-depth) /
                torch.sqrt(uncertainty+1e-10))[mask].sum()

        # 如果启用了颜色loss，那么将会计算一个color_loss，为预测颜色和真实颜色之间的差异；
        if self.use_color_in_tracking:
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum()
            loss += self.w_color_loss*color_loss

        # 触发反向传播，计算梯度；
        loss.backward()
        optimizer.step()
        # 梯度归零，为下一次更新做准备；
        optimizer.zero_grad()
        # 返回计算出的loss;
        return loss.item()

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.
        （从建图线程读取场景表达参数）
        """
        # 检查当前的mapping索引值和之前保存的mapping索引值之间是否保持一致，如果不一致需要更新两项：解码器decoder和特征网格；
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            # 使用深拷贝的方法更新解码器decoder;
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                # 更新特征网格feature_grid(self.c)
                self.c[key] = val
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)      # 以上条件判断信息并不重要；

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]            # 从第0帧开始；执行当前循环；
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]

            # 三种时间策略，关系到tracking和mapping的时间顺序；
            if self.sync_method == 'strict':
                # strictly mapping and then tracking（严格地执行先mapping后tracking的过程）
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                    pre_c2w = self.estimate_c2w_list[idx-1].to(device)
            elif self.sync_method == 'loose':
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                #（可以先mapping后tracking，当然仍然有一个边界值；）
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.1)
            elif self.sync_method == 'free':
                # pure parallel, if mesh/vis happens may cause inbalance（完全自由的时间顺序）
                pass

            self.update_para_from_mapping()     # 调用update_para_from_mapping()函数读取参数；

            if self.verbose:        # verbose:是否需要输出详细信息；
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:      # 一个可视化的函数，并不重要；
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.vis(idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders)

            else:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                # 如果使用基于恒定速度的假设，则进行如下处理：
                if self.const_speed_assumption and idx-2 >= 0:
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        device).float().inverse()
                    estimated_new_cam_c2w = delta@pre_c2w
                # 如果没有使用基于恒定速度的假设，则使用之前的相机位姿作为当前的位姿；
                else:
                    estimated_new_cam_c2w = pre_c2w

                # 使用get_tensor_from_camera()函数将相机位姿从pose（矩阵）转化为tensor;其逆函数为get_camera_from_tensor();
                camera_tensor = get_tensor_from_camera(
                    estimated_new_cam_c2w.detach())
                if self.seperate_LR:
                    # 是否需要分离学习率，即将旋转和平移的学习分离；
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:]
                    quad = camera_tensor[:4]
                    cam_para_list_quad = [quad]
                    quad = Variable(quad, requires_grad=True)
                    T = Variable(T, requires_grad=True)
                    camera_tensor = torch.cat([quad, T], 0)
                    cam_para_list_T = [T]
                    cam_para_list_quad = [quad]
                    # 使用Adam优化器，得到优化好的相机外参数；并作为参数传入到optimize_cam_in_batch()成员函数中，作为损失函数；
                    optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},
                                                         {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}])
                else:
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor]
                    # 使用Adam优化器，得到优化后的相机外参；同上；
                    optimizer_camera = torch.optim.Adam(
                        cam_para_list, lr=self.cam_lr)

                # 计算估计的相机姿态和真实的相机姿态差值的绝对值，用于输出信息时使用；
                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor).mean().item()
                # candidate_cam_tensor用于在for循环中存储最佳的相机姿态；
                candidate_cam_tensor = None
                # current_min_loss用于存储for循环中最小损失值。初始值设得比较大有利于后续更新赋值；
                current_min_loss = 10000000000.

                # 优化相机位姿的for循环；
                for cam_iter in range(self.num_cam_iters):
                    if self.seperate_LR:
                        camera_tensor = torch.cat([quad, T], 0).to(self.device)

                    # 可视化当前迭代结果；
                    self.visualizer.vis(idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)

                    # 使用optimize_cam_in_batch()成员函数优化相机位姿；
                    loss = self.optimize_cam_in_batch(camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)

                    # 记录初始损失值；
                    if cam_iter == 0:
                        initial_loss = loss

                    # 计算相机姿态误差；
                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            print(f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    # 更新loss；
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.clone().detach()
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device)
                # 更新最优相机姿态；
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
            # 在经历过一整个相机姿态的迭代更新之后，将本次最后更新的位姿结果存储到estimate_c2w_list[]中；供后续的Mapper.py使用；
            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            # 给pre_c2w赋值，用在新相机姿态的更新处；
            pre_c2w = c2w.clone()
            self.idx[0] = idx
            if self.low_gpu_mem:
                torch.cuda.empty_cache()