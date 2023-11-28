import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory

from models.dataset import Dataset
from models.sampler import PointSampler
from models.fields import (
    RenderingNetwork,
    SDFNetwork,
    SingleVarianceNetwork,
    NeRF,
    AnisotropyNetwork
)
from models.attenuation_coefficient import AttenuationCoefficient
from models.renderer import Renderer
from models.util import read_mesh, write_mesh, extract_geometry

logging.getLogger('matplotlib.font_manager').disabled = True

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, max_n_training_images=-1):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.viz_deviation_freq = self.conf.get_int('train.viz_deviation_freq', 0)
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.max_n_training_images = max_n_training_images

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        # optionally initialize a background NeRF network
        self.nerf_outside = None
        renders_background = 'model.point_sampler.n_outside' not in self.conf or self.conf['model.point_sampler.n_outside'] > 0
        if renders_background:
            self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
            params_to_train += list(self.nerf_outside.parameters())
            
        # optionally initialize a layer to learn a spatially varying anisotropy 
        self.anisotropy_network = None
        if 'model.anisotropy_network' in self.conf:
            self.anisotropy_network = AnisotropyNetwork(**self.conf['model.anisotropy_network']).to(self.device)
            params_to_train += list(self.anisotropy_network.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.atten_coeff = AttenuationCoefficient(**self.conf['model.attenuation_coefficient'])
        self.point_sampler = PointSampler(**self.conf['model.point_sampler'])
        self.renderer = Renderer(self.nerf_outside,
                                 self.sdf_network,
                                 self.deviation_network,
                                 self.color_network,
                                 self.anisotropy_network,
                                 self.atten_coeff,
                                 self.point_sampler)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()
    
    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              annealed_anisotropy=self.get_annealed_anisotropy())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()
 
            if self.viz_deviation_freq > 0 and self.iter_step % self.viz_deviation_freq == 0:
                self.visualize_deviation_network()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        if self.max_n_training_images <= 0:
            return torch.randperm(self.dataset.n_images)
        else:
            return torch.randperm(min(self.dataset.n_images, self.max_n_training_images))

    def get_annealed_anisotropy(self):
        # goes from 0 (uniform / isotropic / rough) --> 1 (delta / anisotropic / smooth)
        if self.anneal_end == 0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        if 'nerf' in checkpoint: 
            self.nerf_outside.state_dict(checkpoint['nerf'])
        if 'anisotropy_network_fine' in checkpoint:
            self.anisotropy_network.state_dict(checkpoint['anisotropy_network_fine'])

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        if self.nerf_outside is not None: 
            checkpoint['nerf'] = self.nerf_outside.state_dict()
        if self.anisotropy_network is not None:
            checkpoint['anisotropy_network_fine'] = self.anisotropy_network.state_dict()

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_depth = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              background_rgb=background_rgb,
                                              annealed_anisotropy=self.get_annealed_anisotropy())

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            if feasible('gradients') and feasible('weights'):
                n_samples = self.point_sampler.n_total_samples
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)

            if feasible('depth'):
                out_depth.append(render_out['depth'])

            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
        
        depth_fine = None
        if len(out_depth) > 0:
            depth_img = (np.concatenate(out_depth, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
            depth_img = cv.applyColorMap(depth_img, cv.COLORMAP_MAGMA)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'depth'), exist_ok=True)

        if self.mode == 'render':
            os.makedirs(os.path.join(self.base_exp_dir, 'renders'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                if self.mode == 'render':
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'renders',
                                            'render_{}.png'.format(idx)),
                               img_fine[..., i])
                else:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'validations_fine',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                            np.concatenate([img_fine[..., i],
                                            self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])
        
        if len(out_depth) > 0:
            cv.imwrite(os.path.join(self.base_exp_dir,
                                        'depth',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           depth_img)

    def validate_mesh(self, scale_mesh=True, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        
        query_func = lambda pts: -self.sdf_network.sdf(pts)
        vertices, triangles = extract_geometry(bound_min, bound_max, 
                                               resolution=resolution, 
                                               threshold=threshold,
                                               query_func=query_func)

        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if scale_mesh:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
            name = '{:0>8d}.ply'.format(self.iter_step)
        else:
            name = '{:0>8d}-local.obj'.format(self.iter_step)

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', name))

        logging.info('End')

if __name__ == '__main__':
    print('Running..')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_n_training_images', type=int, default=-1)

    # novel view synthesis
    parser.add_argument('--image_idx', type=int, default=0)
    parser.add_argument('--image_resolution_level', type=int, default=1)

    # mesh extraction
    parser.add_argument('--mesh_resolution', type=int, default=512)
    parser.add_argument('--use_local_scale', default=False, action="store_true")
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.max_n_training_images)

    if args.mode == 'train':
        runner.train()

    elif args.mode == 'render':
        runner.validate_image(idx=args.image_idx, 
                              resolution_level=args.image_resolution_level)
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(scale_mesh=(not args.use_local_scale), 
                             resolution=args.mesh_resolution, 
                             threshold=args.mcube_threshold)
    else:
        raise Exception(f'Mode not recognized: {args.mode}') 