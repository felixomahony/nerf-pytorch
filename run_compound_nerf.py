import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import pickle

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk, encoder_id, appendix_id, save_ins = False):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk], (encoder_id==1), (appendix_id==1), save_ins) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, encoder_id, appendix_id, save_ins = False, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk, encoder_id, appendix_id, save_ins)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, encoder_id, appendix_id, chunk=1024*32, save_ins = False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], encoder_id=encoder_id, appendix_id=appendix_id, save_ins = save_ins, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, encoder_id, appendix_id, save_ins = False, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, encoder_id, appendix_id, chunk, save_ins, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, encoder_id, appendix_id, save_ins = False, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, encoder_id, appendix_id, save_ins, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn, e_id, a_id, s_ins = False : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn, 
                                                                encoder_id = e_id, 
                                                                appendix_id = a_id,
                                                                save_ins = s_ins,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                encoder_id = 0,
                appendix_id = 0,
                save_ins = False,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn, encoder_id, appendix_id, save_ins)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn, encoder_id, appendix_id, save_ins)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='./configs/compound.txt',
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir_00", type=str, default='./data/llff/fern', 
                        help='input data directory for style transfer')
    parser.add_argument("--datadir_10", type=str, default='./data/llff/fern', 
                        help='input data directory for transfer origin')
    parser.add_argument("--datadir_11", type=str, default='./data/llff/fern', 
                        help='input data directory for transfer style target')
    

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K_00 = None
    K_10 = None
    K_11 = None

    if args.dataset_type == 'blender':
        #00
        images_00, poses_00, render_poses_00, hwf_00, i_split_00 = load_blender_data(args.datadir_00, args.half_res, args.testskip)
        print('Loaded blender 00', images_00.shape, render_poses_00.shape, hwf_00, args.datadir_00)
        i_train_00, i_val_00, i_test_00 = i_split_00

        near = 2.
        far = 6.

        if args.white_bkgd:
            images_00 = images_00[...,:3]*images_00[...,-1:] + (1.-images_00[...,-1:])
        else:
            images_00 = images_00[...,:3]

        #10
        images_10, poses_10, render_poses_10, hwf_10, i_split_10 = load_blender_data(args.datadir_10, args.half_res, args.testskip)
        print('Loaded blender 10', images_10.shape, render_poses_10.shape, hwf_10, args.datadir_10)
        i_train_10, i_val_10, i_test_10 = i_split_10

        if args.white_bkgd:
            images_10 = images_10[...,:3]*images_10[...,-1:] + (1.-images_10[...,-1:])
        else:
            images_10 = images_10[...,:3]

        #11
        images_11, poses_11, render_poses_11, hwf_11, i_split_11 = load_blender_data(args.datadir_11, args.half_res, args.testskip)
        print('Loaded blender 11', images_11.shape, render_poses_11.shape, hwf_11, args.datadir_11)
        i_train_11, i_val_11, i_test_11 = i_split_11

        if args.white_bkgd:
            images_11 = images_11[...,:3]*images_11[...,-1:] + (1.-images_11[...,-1:])
        else:
            images_11 = images_11[...,:3]

        
    ##00
    # Cast intrinsics to right types
    H_00, W_00, focal_00 = hwf_00
    H_00, W_00 = int(H_00), int(W_00)
    hwf_00 = [H_00, W_00, focal_00]

    if K_00 is None:
        K_00 = np.array([
            [focal_00, 0, 0.5*W_00],
            [0, focal_00, 0.5*H_00],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses_00 = np.array(poses_00[i_test_00])

    ##10
    # Cast intrinsics to right types
    H_10, W_10, focal_10 = hwf_10
    H_10, W_10 = int(H_10), int(W_10)
    hwf_10 = [H_10, W_10, focal_10]

    if K_10 is None:
        K_10 = np.array([
            [focal_10, 0, 0.5*W_10],
            [0, focal_10, 0.5*H_10],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses_10 = np.array(poses_10[i_test_10])

    ##00
    # Cast intrinsics to right types
    H_11, W_11, focal_11 = hwf_11
    H_11, W_11 = int(H_11), int(W_11)
    hwf_11 = [H_00, W_11, focal_11]

    if K_11 is None:
        K_11 = np.array([
            [focal_11, 0, 0.5*W_11],
            [0, focal_11, 0.5*H_11],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses_11 = np.array(poses_11[i_test_11])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses_00 = torch.Tensor(render_poses_00).to(device)
    # Move testing data to GPU
    render_poses_00 = torch.Tensor(render_poses_00).to(device)
    # Move testing data to GPU
    render_poses_00 = torch.Tensor(render_poses_00).to(device)


    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        #00
        # For random ray batching
        print('get rays')
        rays_00 = np.stack([get_rays_np(H_00, W_00, K_00, p) for p in poses_00[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb_00 = np.concatenate([rays_00, images_00[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb_00 = np.transpose(rays_rgb_00, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb_00 = np.stack([rays_rgb_00[i] for i in i_train_00], 0) # train images only
        rays_rgb_00 = np.reshape(rays_rgb_00, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb_00 = rays_rgb_00.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb_00)

        #10
        # For random ray batching
        print('get rays')
        rays_10 = np.stack([get_rays_np(H_10, W_10, K_10, p) for p in poses_10[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb_10 = np.concatenate([rays_10, images_10[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb_10 = np.transpose(rays_rgb_10, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb_10 = np.stack([rays_rgb_10[i] for i in i_train_10], 0) # train images only
        rays_rgb_10 = np.reshape(rays_rgb_10, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb_10 = rays_rgb_10.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb_10)

        #11
        # For random ray batching
        print('get rays')
        rays_11 = np.stack([get_rays_np(H_11, W_11, K_11, p) for p in poses_11[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb_11 = np.concatenate([rays_11, images_11[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb_11 = np.transpose(rays_rgb_11, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb_11 = np.stack([rays_rgb_11[i] for i in i_train_11], 0) # train images only
        rays_rgb_11 = np.reshape(rays_rgb_11, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb_11 = rays_rgb_11.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb_11)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images_00 = torch.Tensor(images_00).to(device)
        images_10 = torch.Tensor(images_10).to(device)
        images_11 = torch.Tensor(images_11).to(device)
    poses_00 = torch.Tensor(poses_00).to(device)
    poses_10 = torch.Tensor(poses_10).to(device)
    poses_11 = torch.Tensor(poses_11).to(device)
    if use_batching:
        rays_rgb_00 = torch.Tensor(rays_rgb_00).to(device)
        rays_rgb_10 = torch.Tensor(rays_rgb_10).to(device)
        rays_rgb_11 = torch.Tensor(rays_rgb_11).to(device)


    N_iters = 200000 + 1
    print('Begin')
    print('------Scene 00-----')
    print('TRAIN views are', i_train_00)
    print('TEST views are', i_test_00)
    print('VAL views are', i_val_00)
    print('------Scene 10-----')
    print('TRAIN views are', i_train_10)
    print('TEST views are', i_test_10)
    print('VAL views are', i_val_10)
    print('------Scene 11-----')
    print('TRAIN views are', i_train_11)
    print('TEST views are', i_test_11)
    print('VAL views are', i_val_11)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            #don't permit batching because of time constraints
            pass
            # # Random over all images
            # batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            # batch = torch.transpose(batch, 0, 1)
            # batch_rays, target_s = batch[:2], batch[2]

            # i_batch += N_rand
            # if i_batch >= rays_rgb.shape[0]:
            #     print("Shuffle data after an epoch!")
            #     rand_idx = torch.randperm(rays_rgb.shape[0])
            #     rays_rgb = rays_rgb[rand_idx]
            #     i_batch = 0

        else:
            #00
            # Random from one image
            # print("00")
            img_i_00 = np.random.choice(i_train_00)
            target_00 = images_00[img_i_00]
            target_00 = torch.Tensor(target_00).to(device)
            pose_00 = poses_00[img_i_00, :3,:4]

            if N_rand is not None:
                rays_o_00, rays_d_00 = get_rays(H_00, W_00, K_00, torch.Tensor(pose_00))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H_00//2 * args.precrop_frac)
                    dW = int(W_00//2 * args.precrop_frac)
                    coords_00 = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H_00//2 - dH, H_00//2 + dH - 1, 2*dH), 
                            torch.linspace(W_00//2 - dW, W_00//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords_00 = torch.stack(torch.meshgrid(torch.linspace(0, H_00-1, H_00), torch.linspace(0, W_00-1, W_00)), -1)  # (H, W, 2)

                coords_00 = torch.reshape(coords_00, [-1,2])  # (H * W, 2)
                select_inds_00 = np.random.choice(coords_00.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords_00 = coords_00[select_inds_00].long()  # (N_rand, 2)
                rays_o_00 = rays_o_00[select_coords_00[:, 0], select_coords_00[:, 1]]  # (N_rand, 3)
                rays_d_00 = rays_d_00[select_coords_00[:, 0], select_coords_00[:, 1]]  # (N_rand, 3)
                batch_rays_00 = torch.stack([rays_o_00, rays_d_00], 0)
                target_s_00 = target_00[select_coords_00[:, 0], select_coords_00[:, 1]]  # (N_rand, 3)
            
            #10
            # Random from one image
            # print("10")
            img_i_10 = np.random.choice(i_train_10)
            target_10 = images_10[img_i_10]
            target_10 = torch.Tensor(target_10).to(device)
            pose_10 = poses_10[img_i_10, :3,:4]

            if N_rand is not None:
                rays_o_10, rays_d_10 = get_rays(H_10, W_10, K_10, torch.Tensor(pose_10))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H_10//2 * args.precrop_frac)
                    dW = int(W_10//2 * args.precrop_frac)
                    coords_10 = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H_10//2 - dH, H_10//2 + dH - 1, 2*dH), 
                            torch.linspace(W_10//2 - dW, W_10//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords_10 = torch.stack(torch.meshgrid(torch.linspace(0, H_10-1, H_10), torch.linspace(0, W_10-1, W_10)), -1)  # (H, W, 2)

                coords_10 = torch.reshape(coords_10, [-1,2])  # (H * W, 2)
                select_inds_10 = np.random.choice(coords_10.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords_10 = coords_10[select_inds_10].long()  # (N_rand, 2)
                rays_o_10 = rays_o_10[select_coords_10[:, 0], select_coords_10[:, 1]]  # (N_rand, 3)
                rays_d_10 = rays_d_10[select_coords_10[:, 0], select_coords_10[:, 1]]  # (N_rand, 3)
                batch_rays_10 = torch.stack([rays_o_10, rays_d_10], 0)
                target_s_10 = target_10[select_coords_10[:, 0], select_coords_10[:, 1]]  # (N_rand, 3)
            
            #11
            # Random from one image
            # print("11")
            img_i_11 = np.random.choice(i_train_11)
            target_11 = images_11[img_i_11]
            target_11 = torch.Tensor(target_11).to(device)
            pose_11 = poses_11[img_i_11, :3,:4]

            if N_rand is not None:
                rays_o_11, rays_d_11 = get_rays(H_11, W_11, K_11, torch.Tensor(pose_11))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H_11//2 * args.precrop_frac)
                    dW = int(W_11//2 * args.precrop_frac)
                    coords_11 = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H_11//2 - dH, H_11//2 + dH - 1, 2*dH), 
                            torch.linspace(W_11//2 - dW, W_11//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords_11 = torch.stack(torch.meshgrid(torch.linspace(0, H_11-1, H_11), torch.linspace(0, W_11-1, W_11)), -1)  # (H, W, 2)

                coords_11 = torch.reshape(coords_11, [-1,2])  # (H * W, 2)
                select_inds_11 = np.random.choice(coords_11.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords_11 = coords_11[select_inds_11].long()  # (N_rand, 2)
                rays_o_11 = rays_o_11[select_coords_11[:, 0], select_coords_11[:, 1]]  # (N_rand, 3)
                rays_d_11 = rays_d_11[select_coords_11[:, 0], select_coords_11[:, 1]]  # (N_rand, 3)
                batch_rays_11 = torch.stack([rays_o_11, rays_d_11], 0)
                target_s_11 = target_11[select_coords_11[:, 0], select_coords_11[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb_00, disp_00, acc_00, extras_00 = render(H_00, W_00, K_00, 0, 0, chunk=args.chunk, rays=batch_rays_00,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        
        #####  Core optimization loop  #####
        rgb_10, disp_10, acc_10, extras_10 = render(H_10, W_10, K_10, 1, 0, chunk=args.chunk, rays=batch_rays_10,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        
        #####  Core optimization loop  #####
        rgb_11, disp_11, acc_11, extras_11 = render(H_11, W_11, K_11, 1, 1, chunk=args.chunk, rays=batch_rays_11,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss_00 = img2mse(rgb_00, target_s_00)
        trans_00 = extras_00['raw'][...,-1]
        loss_00 = img_loss_00
        psnr_00 = mse2psnr(img_loss_00)

        if 'rgb0' in extras_00:
            img_loss0_00 = img2mse(extras_00['rgb0'], target_s_00)
            loss_00 = loss_00 + img_loss0_00
            psnr0_00 = mse2psnr(img_loss0_00)

        # loss.backward()
        # optimizer.step()

    
        # optimizer.zero_grad()
        img_loss_10 = img2mse(rgb_10, target_s_10)
        trans_10 = extras_10['raw'][...,-1]
        loss_10 = img_loss_10
        psnr_10 = mse2psnr(img_loss_10)

        if 'rgb0' in extras_10:
            img_loss0_10 = img2mse(extras_10['rgb0'], target_s_10)
            loss_10 = loss_10 + img_loss0_10
            psnr0_10 = mse2psnr(img_loss0_10)

        # loss.backward()
        # optimizer.step()


        # optimizer.zero_grad()
        img_loss_11 = img2mse(rgb_11, target_s_11)
        trans_11 = extras_11['raw'][...,-1]
        loss_11 = img_loss_11
        psnr_11 = mse2psnr(img_loss_11)

        if 'rgb0' in extras_11:
            img_loss0_11 = img2mse(extras_11['rgb0'], target_s_11)
            loss_11 = loss_11 + img_loss0_11
            psnr0_11 = mse2psnr(img_loss0_11)

        loss = loss_00+loss_10+loss_11
        loss.backward()
        optimizer.step()

        

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            path_pickle = os.path.join(basedir, expname, '{:06d}.pkl'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': (None if render_kwargs_train['network_fine'] is None else render_kwargs_train['network_fine'].state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            # dvc = render_kwargs_train['network_fn'].get_device()
            with open(path_pickle, "wb") as f:
                pickle.dump((render_kwargs_train['network_fn'], render_kwargs_train['network_fine'],), f)
            # render_kwargs_train['network_fn'].to(device)
            # render_kwargs_train['network_fine'].to(device)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses_00, hwf_00, K_00, args.chunk, render_kwargs_test, 0, 0, True)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb_00.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp_00.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses_10, hwf_10, K_10, args.chunk, render_kwargs_test, 1, 0)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb_10.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp_10.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses_11, hwf_11, K_11, args.chunk, render_kwargs_test, 1, 1)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb_11.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp_11.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses_00, hwf_00, K_00, args.chunk, render_kwargs_test, 0, 1)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb_01.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp_01.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}_00'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses_00[i_test_00].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses_00[i_test_00]).to(device), hwf_00, K_00, args.chunk, render_kwargs_test, 0, 0, gt_imgs=images_00[i_test_00], savedir=testsavedir)

            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}_10'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                render_path(torch.Tensor(poses_10[i_test_10]).to(device), hwf_10, K_10, args.chunk, render_kwargs_test, 1, 0, gt_imgs=images_00[i_test_10], savedir=testsavedir)

            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}_11'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                render_path(torch.Tensor(poses_11[i_test_11]).to(device), hwf_11, K_11, args.chunk, render_kwargs_test, 1, 1, gt_imgs=images_11[i_test_11], savedir=testsavedir)

            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}_01'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                render_path(torch.Tensor(poses_00[i_test_00]).to(device), hwf_00, K_00, args.chunk, render_kwargs_test, 0, 1, gt_imgs=images_00[i_test_00], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss_00: {loss_00.item()}  PSNR_00: {psnr_00.item()}")
            tqdm.write(f"[TRAIN] Iter: {i} Loss_10: {loss_10.item()}  PSNR_10: {psnr_10.item()}")
            tqdm.write(f"[TRAIN] Iter: {i} Loss_11: {loss_11.item()}  PSNR_11: {psnr_11.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
