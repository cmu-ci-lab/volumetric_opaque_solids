import math
import torch
import torch.nn.functional as F

EPSILON = 1e-6
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
INV_SQRT_2 = 1.0 / math.sqrt(2.0)
INV_SQRT_2_PI = 1.0 / math.sqrt(2.0 * math.pi)

def lerp(a, b, gamma):
    return a * (1.0 - gamma) + b * gamma

def compute_cos(ray_dir, grad_mean_implicit):
    implicit_normal = -F.normalize(grad_mean_implicit)
    return (ray_dir * implicit_normal).sum(-1, keepdim=True)

class ProjectedArea:  
    @staticmethod
    def uniform(ray_dir, grad_mean_implicit, alpha):
        return 0.5

    @staticmethod
    def delta(ray_dir, grad_mean_implicit, alpha):
        return torch.abs(compute_cos(ray_dir, grad_mean_implicit))

    @staticmethod
    def delta_relu(ray_dir, grad_mean_implicit, alpha):
        return F.relu(compute_cos(ray_dir, grad_mean_implicit))

    @staticmethod
    def linear_mixture(ray_dir, grad_mean_implicit, alpha):
        # alpha := mixture / anisotropy parameter.
        # alpha == 0 --> uniform
        # alpha == 1 --> delta
        assert alpha is not None, 'alpha must be defined to use mixture based projected area.'
        abs_cos = torch.abs(compute_cos(ray_dir, grad_mean_implicit))
        return lerp(0.5, abs_cos, alpha)

    @staticmethod
    def linear_mixture_relu(ray_dir, grad_mean_implicit, alpha):
        # alpha := cosine annealing parameter.
        assert alpha is not None, 'alpha must be defined to use NeuS based projected area.'
        # Adapted from NeuS codebase: https://github.com/Totoro97/NeuS/blob/6f96f96005d72a7a358379d2b576c496a1ab68dd/models/renderer.py#L230-L236 
        cos = compute_cos(ray_dir, grad_mean_implicit)
        return lerp(F.relu(cos * 0.5 + 0.5), F.relu(cos), alpha)

    @staticmethod
    def sggx(ray_dir, grad_mean_implicit, alpha):
        alpha_squared = alpha * alpha
        cos = compute_cos(ray_dir, grad_mean_implicit)
        cos_squared = cos * cos
        unnormalized_projected_area = torch.sqrt(cos_squared * alpha_squared + (1.0 - alpha_squared))
        sinh_arg = (alpha / torch.sqrt(1.0 - alpha_squared))
        normalization = 1.0 + (1.0 - alpha_squared) * torch.asinh(sinh_arg)
        return unnormalized_projected_area / normalization

    @staticmethod
    def get(normal_distribution):
        if normal_distribution == 'delta':
            return ProjectedArea.delta
        elif normal_distribution == 'delta_relu':
            return ProjectedArea.delta_relu
        elif normal_distribution == 'uniform':
            return ProjectedArea.uniform
        elif normal_distribution == 'linear_mixture':
            return ProjectedArea.linear_mixture
        elif normal_distribution == 'linear_mixture_relu':
            return ProjectedArea.linear_mixture_relu
        elif normal_distribution == 'sggx':
            return ProjectedArea.sggx
        else:
            raise Exception(f'Unsupported projected area type {normal_distribution}')

class Density:
    @staticmethod
    def logistic(mean_implicit, inv_std):
        # pdf = (1.0 - sigmoid(mean_implicit * inv_std)) sigmoid(mean_implicit * inv_std) * inv_std
        # cdf = sigmoid(x * inv_std)
        # pdf / cdf = (1.0 - sigmoid(mean_implicit * inv_std)) * inv_std = inv_std * sigmoid(-mean_implicit * inv_std)
        return inv_std * torch.sigmoid(-mean_implicit * inv_std)

    @staticmethod
    def gaussian(mean_implicit, inv_std):
        pdf = inv_std * INV_SQRT_2_PI * torch.exp(-0.5 * torch.square(mean_implicit * inv_std))
        cdf = 0.5 + 0.5 * torch.erf(mean_implicit * inv_std * INV_SQRT_2)
        return pdf / (cdf + EPSILON)

    @staticmethod
    def laplace(mean_implicit, inv_std):
        exp_term = torch.exp(-torch.abs(mean_implicit) * inv_std)
        pdf = inv_std * 0.5 * exp_term
        cdf = 0.5 + 0.5 * torch.sign(mean_implicit) * (1.0 - exp_term)
        return pdf / (cdf + EPSILON)

    @staticmethod
    def get(implicit_distribution):
        if implicit_distribution == 'logistic':
            return Density.logistic
        elif implicit_distribution == 'gaussian':
            return Density.gaussian
        elif implicit_distribution == 'laplace':
            return Density.laplace
        else:
            raise Exception(f'Unsupported density type {implicit_distribution}')

class AttenuationCoefficient:
    def __init__(self, 
                 implicit_distribution = 'gaussian', 
                 normal_distribution = 'linear_mixture'):
        self.implicit_distribution = implicit_distribution
        self.normal_distribution = normal_distribution
        self.density = Density.get(implicit_distribution)
        self.projected_area = ProjectedArea.get(normal_distribution)

    def __call__(self, ray_dir, mean_implicit, grad_mean_implicit, inv_std, anisotropy_param):
        sigma_perp = self.projected_area(ray_dir, grad_mean_implicit, anisotropy_param)
        sigma_parallel = self.density(mean_implicit, inv_std)
        return sigma_perp * sigma_parallel
