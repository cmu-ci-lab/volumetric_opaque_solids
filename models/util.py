import torch
import mcubes
import numpy as np

def read_mesh(path, quad = False):
    V = []
    F = []
    with open(path) as file:
        for line in file:
            tokens = line.strip('\n').split(' ')
            if tokens[0] == 'v':
                V.append(np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])]))
            
            if tokens[0] == 'f':
                if quad:
                    F.append(np.array([int(tokens[1]), int(tokens[2]), int(tokens[3]), int(tokens[4])]))
                else:
                    F.append(np.array([int(tokens[1]), int(tokens[2]), int(tokens[3])]))

    return np.array(V), np.array(F)

def write_mesh(path, vertices, faces, data, quad = False):
    with open(path, 'w') as out:
        out.write('# OBJ file\n')

        for i in range(vertices.shape[0]):
            out.write('v {:.8f} {:.8f} {:.8f} \n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))

        for i in range(data.shape[0]):
            out.write('vt {:.8f} 0 \n'.format(data[i]))

        for i in range(faces.shape[0]):
            fi = faces[i, 0]
            fj = faces[i, 1]
            fk = faces[i, 2]
            if quad:
                fl = faces[i, 3]
                out.write('f {:d}/{:d} {:d}/{:d} {:d}/{:d} {:d}/{:d}\n'.format(fi, fi, fj, fj, fk, fk, fl, fl))
            else:
                out.write('f {:d}/{:d} {:d}/{:d} {:d}/{:d}\n'.format(fi, fi, fj, fj, fk, fk))

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()
    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles
