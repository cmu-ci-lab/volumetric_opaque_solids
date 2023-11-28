import torch
import numpy as np
import time

class PointSampler:
    def __init__(self,
                 n_sdf_pts = 1024,
                 n_fg_samples = 28,
                 n_surf_samples = 8,
                 n_bg_samples = 28,
                 n_outside = 32,
                 use_random_binary_search = False,
                 use_sdf_offset = False):
        # number of initial evaluations of sdf along each ray
        self.n_sdf_pts = n_sdf_pts
        self.n_sdf_samples = 10

        # number of points sampled per ray in the foreground, surface interval, and background
        self.n_fg_samples = n_fg_samples
        self.n_surf_samples = n_surf_samples
        self.n_bg_samples = n_bg_samples

        # total number of (primary, non-background) samples along each ray
        self.n_total_samples = n_fg_samples + n_surf_samples + n_bg_samples

        # number of points sampled per ray in background
        self.n_outside = n_outside

        self.use_random_binary_search = use_random_binary_search
        self.use_sdf_offset = use_sdf_offset

    def eval_at_points(self, rays_o, rays_d, depth, f):
        pts = rays_o[:, None, :] + rays_d[:, None, :] * depth[:, :, None]
        with torch.no_grad():
            val = f(pts.reshape(-1, 3)).reshape(depth.shape[0], depth.shape[1]).squeeze(dim=-1)
        return val

    def sample_interval_with_random_binary_search(self, num_samples, start, stop, rays_o, rays_d, f):
        '''
        Performs a random binary search for the x such that f(x) = 0 given f(start) > 0 and f(stop) < 0
        returns the entire sequence of sampled points, sorted from smallest to largest z val.
        '''
        current_min, current_max = start, stop
        samples = torch.zeros((start.shape[0], num_samples))
        uniform_random = torch.rand(samples.shape)
        for i in range(num_samples):
            samples[:, i] = (current_max - current_min) * uniform_random[:, i] + current_min
            f_val = self.eval_at_points(rays_o, rays_d, samples[:, i].unsqueeze(dim=1), f)
            current_min = torch.where(f_val <= 0, current_min, samples[:, i])
            current_max = torch.where(f_val <= 0, samples[:, i], current_max)
        return torch.sort(samples)[0]
    
    def sample_interval_uniformly(self, n, start, stop):
        start = start if len(start.shape) == 1 else start.squeeze(dim=-1)
        stop = stop if len(stop.shape) == 1 else stop.squeeze(dim=-1)
        x = torch.linspace(0, 1.0 - 1.0 / n, n)[None, :]
        x = x * (stop - start)[:, None] + start[:, None]
        x += (torch.rand(start.shape[0]) * (stop - start) / n)[:, None]
        return x

    def _dense_sdf_evaluation(self, rays_o, rays_d, near, far, sdf_func):
        uniform_z = torch.linspace(0.0, 1.0, self.n_sdf_pts + 1)
        z = near + (far - near) * uniform_z[None, :]
        return z, self.eval_at_points(rays_o, rays_d, z, sdf_func)

    def _find_first_zero_crossing(self, sdf):
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        sign_change = (next_sdf * prev_sdf < 0).long()
        return sign_change.argmax(1).long()

    def _compute_surface_z_bound(self, isect_idx, z, near, far):
        z_bounds = torch.gather(z, dim=1, index=torch.cat([isect_idx[:, None], isect_idx[:, None]+1], dim=1)).squeeze(dim=-1)
        return z_bounds[:, 0], z_bounds[:, 1]

    def sample_intersection(self, rays_o, rays_d, near, far, sdf_func, inv_std):
        with torch.no_grad():
            z, sdf = self._dense_sdf_evaluation(rays_o, rays_d, near, far, sdf_func)
            if self.use_sdf_offset:
                sdf += torch.normal(0, 1.0 / inv_std)

        isect_idx = self._find_first_zero_crossing(sdf) 
        surf_lower, surf_upper = self._compute_surface_z_bound(isect_idx, z, near, far)
        
        has_isect = (isect_idx > 0).bool()
        no_isect = torch.logical_not(has_isect)

        # final depth samples buffers
        z_vals = torch.empty((rays_o.shape[0], self.n_total_samples))

        # depth map for visualization
        surf_z_image = torch.zeros_like(rays_o)

        if torch.any(has_isect):
            fg_z = self.sample_interval_uniformly(self.n_fg_samples, near[has_isect], surf_lower[has_isect])
            bg_z = self.sample_interval_uniformly(self.n_bg_samples, surf_upper[has_isect], far[has_isect])
            if not self.use_random_binary_search:
                surf_z = self.sample_interval_uniformly(self.n_surf_samples, surf_lower[has_isect], surf_upper[has_isect])
            else:
                surf_z = self.sample_interval_with_random_binary_search(self.n_surf_samples,
                                                                        surf_lower[has_isect],
                                                                        surf_upper[has_isect],
                                                                        rays_o[has_isect],
                                                                        rays_d[has_isect],
                                                                        sdf_func)
            z_vals[has_isect, :] = torch.cat([fg_z, surf_z, bg_z], dim=-1)	
            
            # return z-val in image for debugging
            surf_lower_unit_z = (surf_lower - near.squeeze()) / (far - near).squeeze()
            surf_z_image[has_isect, :] = surf_lower_unit_z[has_isect, None].repeat(1, 3)

        if torch.any(no_isect):
            z_vals[no_isect, :] = self.sample_interval_uniformly(self.n_total_samples, near[no_isect], far[no_isect])
        
        return z_vals, surf_z_image 

    def sample_outside(self, rays_o, rays_d, far):	
        # Same as NeuS: https://github.com/Totoro97/NeuS/blob/6f96f96005d72a7a358379d2b576c496a1ab68dd/models/renderer.py#L292C19-L313
        if self.n_outside == 0:
            return None
        batch_size = len(rays_o)
        z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)
        mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
        upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
        lower = torch.cat([z_vals_outside[..., :1], mids], -1)
        t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
        z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand
        z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_total_samples
        return z_vals_outside
