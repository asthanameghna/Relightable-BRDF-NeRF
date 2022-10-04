import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

import sys
import tensorflow as tf
import numpy as np
import imageio
import json
import random
import time
from load_llff import load_llff_data

import relight_brdf_nerf

##################### This file renders 360-deg shadows.png files for any trained model - both multipose+single_lightdir AND singlepose+multi_lightdir ###################

architecture = relight_brdf_nerf
##### Multiple pre-trained folders ######
# expname = 'exp267_cowPNG'
# expname = 'exp251_buddhaPNG'
# expname = 'exp205_readingPNG'
# expname = 'exp206_pot2PNG'
expname = 'exp204_bearPNG'
model_no = 400000


tf.compat.v1.enable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
print('############## Allowing Growth ###########')

def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)


tf.compat.v1.keras.backend.set_session(tf.Session(config=config))


basedir = './logs'



config = os.path.join(basedir, expname, 'config.txt')
print('Args:')
print(open(config, 'r').read())

parser = architecture.config_parser()
ft_str = ''
ft_str = '--ft_path {}'.format(os.path.join(basedir, expname, 'model_{:06d}.npy'.format(model_no)))
args = parser.parse_args('--config {} '.format(config) + ft_str)

images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, factor=args.factor,
                                                          recenter=False, bd_factor=.75,
                                                          spherify=True)

hwf = poses[0, :3, -1]
poses = poses[:, :3, :4]
print('Loaded llff', images.shape,
      render_poses.shape, hwf, args.datadir)
if not isinstance(i_test, list):
    i_test = [i_test]

if args.llffhold > 0:
    print('Auto LLFF holdout,', args.llffhold)
    i_test = np.arange(images.shape[0])[::args.llffhold]

i_val = i_test
i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                    (i not in i_test and i not in i_val)])

print('DEFINING BOUNDS')
if args.no_ndc:
    near = tf.reduce_min(bds) * .9
    far = tf.reduce_max(bds) * 1.
else:
    near = 0.
    far = 1.
print('NEAR FAR', near, far)

light_dirs = np.load(args.lightdirsdir, )

# Cast intrinsics to right types
H, W, focal = hwf
H, W = int(H), int(W)
hwf = [H, W, focal]

render_kwargs_train, render_kwargs_test, start, grad_vars, models = architecture.create_nerf_relight(args)

bds_dict = {
    'near': tf.cast(near, tf.float32),
    'far': tf.cast(far, tf.float32),
}
render_kwargs_train.update(bds_dict)
render_kwargs_test.update(bds_dict)

N_rand = args.N_rand
i =0

def render_rays(ray_batch,
                network_fn,
                network_query_fn_relight,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                render_output_type='rgb',
                verbose=False, **kwargs):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, light direction, min
        dist, max dist, and unit-magnitude viewing direction, unit-magnitude lighting direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn_relight: function used for passing queries to network_fn_relight.
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

    def raw2outputs(raw, z_vals, rays_d):
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

        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
                                                             tf.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract RGB of each sample position along each ray.
        # rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # rgb = raw[..., :3]
        # n_norm = raw[...,4:7]
        # bottleneck = raw[...,7:10]
        shadow_ch1 = raw[...,-1]

        shadow = tf.stack([shadow_ch1,shadow_ch1,shadow_ch1], axis=-1)


        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        eps = 1e-05
        alpha_loss = tf.math.log(alpha+eps) + tf.math.log(1-alpha+eps)

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * \
                  tf.math.cumprod(1. - alpha + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(
            weights[..., None] * shadow, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        disp_map = 1. / tf.maximum(1e-10, depth_map /
                                   tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map, alpha_loss

    ###############################
    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_o, rays_d, light_d = ray_batch[:, 0:3], ray_batch[:, 3:6], ray_batch[:, 6:9]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, 3:6] if ray_batch.shape[-1] > 8 else None  # might need to remove if-statement

    # Extract unit-normalized viewing direction.
    lightdirs = ray_batch[:, 6:9] if ray_batch.shape[-1] > 8 else None  # might need to remove if-statement

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 9:11], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = tf.linspace(0., 1., N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
          z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # Evaluate model at each point.
    # raw = network_query_fn(pts, viewdirs, network_fn)  # [N_rays, N_samples, 4]
    raw = network_query_fn_relight(pts, viewdirs, lightdirs, network_fn)  # [N_rays, N_samples, 4]

    rgb_map, disp_map, acc_map, weights, depth_map, alpha_loss = raw2outputs(
        raw, z_vals, rays_d)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # Obtain additional integration times to evaluate based on the weights
        # assigned to colors in the coarse model.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = architecture.sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)

        # Obtain all points to evaluate color, density at.
        z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        # Make predictions with network_fine.
        run_fn = network_fn if network_fine is None else network_fine
        # raw = network_query_fn(pts, viewdirs, run_fn)
        raw = network_query_fn_relight(pts, viewdirs, lightdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map, alpha_loss = raw2outputs(
            raw, z_vals, rays_d)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['alpha_loss'] = alpha_loss
        ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret

def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render_relight(H, W, focal, lightdir=None,
                   chunk=1024 * 32, rays=None, c2w=None, ndc=True,
                   near=0., far=1.,
                   use_viewdirs=False, use_lightdirs=False, c2w_staticcam=None,
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
      light_dir: array of shape [3, 1]. Light direction matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      use_lightdirs: bool. If True, use lighting direction of a point in space in model.
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
        rays_o, rays_d, light_d = architecture.get_rays_relight(H, W, focal, c2w, lightdir)  # kinda fixed
        # light_d = # same size as ray_0 //replicate lightdirs only one image rendered here
    else:
        # use provided ray batch
        rays_o, rays_d, light_d = rays

    if (use_viewdirs == True) and (use_lightdirs == True):
        # provide ray directions and lighting directions as input
        viewdirs = rays_d
        lightdirs = light_d
        # Not changing as it is not used --- in relighting
        # if c2w_staticcam is not None:
        #         #     # special case to visualize effect of viewdirs and lightdirs
        #         #     rays_o, rays_d, light_d = get_rays_relight(H, W, focal, c2w_staticcam, lightdirs)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)
        lightdirs = lightdirs / tf.linalg.norm(lightdirs, axis=-1, keepdims=True)
        lightdirs = tf.cast(tf.reshape(lightdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = architecture.ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    light_d = tf.cast(tf.reshape(light_d, [-1, 3]), dtype=tf.float32)

    near, far = near * \
                tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, light direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, light_d, near, far], axis=-1)

    if (use_viewdirs == True) and (use_lightdirs == True):
        # (ray origin, ray direction, light direction, min dist, max dist, normalized viewing direction, normalized lighting direction)
        rays = tf.concat([rays, viewdirs], axis=-1)
        rays = tf.concat([rays, lightdirs], axis=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]



def render_relight_path_multilight_onepose(render_pose, lightdirs, hwf, chunk, render_kwargs, gt_imgs=None,
                                           savedir=None, render_factor=0, **kwargs):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, l in enumerate(lightdirs):
        print(i, time.time() - t)
        t = time.time()

        rgb, disp, acc, _ = render_relight(
            H, W, focal, l, chunk=chunk, c2w=render_pose[:3, :4], **render_kwargs)

        gam = tf.convert_to_tensor(tf.constant([1 / 2.2]), dtype=tf.float32)
        rgb = tf.math.pow(rgb, gam)

        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps



# Dimensions of Rendered Image
H = 512.0
W = 612.0

angles = []

# Save out the validation image for Tensorboard-free monitoring
testimgdir = os.path.join(basedir, expname, 'poses_shadows')
os.makedirs(testimgdir, exist_ok=True)

print('H ',H)
print('W ',W)

for i in range(0,20):
    img_i = i * 32
    target = images[img_i]
    pose = poses[img_i, :3, :4]


    rgb, disp, acc, extras = render_relight(H, W, focal, args.lightdir, chunk=args.chunk, c2w=pose,
                                            **render_kwargs_test)


    shadow = rgb
    saveimg_dir_r = os.path.join(
            basedir, expname, 'poses_shadows/render_p{:02d}.png'.format(i + 1))
    imageio.imwrite(saveimg_dir_r, architecture.to8b(shadow))


print('DONE!')