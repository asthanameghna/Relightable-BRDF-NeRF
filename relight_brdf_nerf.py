
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import tensorflow as tf
import numpy as np
import imageio
import json
import random
import time
from load_llff import load_llff_data

tf.compat.v1.enable_eager_execution()


# Misc utils

def img2mse(x, y): return tf.reduce_mean(tf.square(x - y))


def mse2psnr(x): return -10. * tf.log(x) / tf.log(10.)


def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def init_nerf_relight_model(D=8, W=256, input_ch=3, input_ch_views=3, input_ch_lights=3, output_ch=4, skips=[4],
                            use_viewdirs=False, use_lightdirs=False):
    relu = tf.keras.layers.ReLU()

    def dense(W=1, act=relu, name='name'):
        return tf.keras.layers.Dense(W, activation=act, name=name)

    def print_layer(layer, name, message): 
        return tf.keras.layers.Lambda(
            (lambda x: tf.Print(x, [x[1005:1010, :]], message=message, first_n=-1, summarize=1024)), name=name)(layer)

    ######## Relightable BRDF NeRF Architecture (as explained in the paper) ###################

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)
    input_ch_lights = int(input_ch_lights)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views + input_ch_lights), name='input_channels')
    inputs_pts, inputs_views, inputs_lights = tf.split(inputs, [input_ch, input_ch_views, input_ch_lights], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])
    inputs_lights.set_shape([None, input_ch_lights])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape, inputs_lights.shape)
    outputs = inputs_pts
    for i in range(D):
        outputs = dense(W, name='Dense_{}'.format(i))(outputs)
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)

    if (use_viewdirs == True) and (use_lightdirs == False):
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs

        for i in range(4):  # changed from 1
            outputs = dense(W // 2)(outputs)
        outputs = dense(3, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)

    elif (use_viewdirs == True) and (use_lightdirs == True):

        alpha_out = dense(1, act=None, name='alpha_out')(outputs)

        n = dense(3, act=None, name='n')(outputs)
        n_norm = n / tf.linalg.norm(n, axis=-1, keepdims=True)

        half = tf.math.add(inputs_views, inputs_lights, name='half')
        half_norm = half / tf.linalg.norm(half, axis=-1, keepdims=True)

        ##### Lambertian ####
        albedo = dense(3, act=tf.keras.activations.sigmoid, name='albedo')(outputs)

        dot_lights = tf.reduce_sum(n_norm * inputs_lights, -1, keepdims=True)
        theta_i = tf.math.acos(dot_lights)
        dot_lights_noneg = tf.nn.relu(dot_lights)

        dot_half = tf.reduce_sum(n_norm * half_norm, -1, keepdims=True)
        theta_h = tf.math.acos(dot_half)
        dot_half_noneg = tf.nn.relu(dot_half)


        ##### Learnable BRDF ######
        brdf_params = dense(3, act=None, name='brdf_params')(outputs)  # for the new version I changed back to 3 from 12

        inputs_2 = tf.concat(
            [brdf_params, dot_lights_noneg, dot_half_noneg], -1)  # concat theta_i, theta_h
        outputs_2 = inputs_2

        for i in range(2): 
            # my original iteration of neural brdf used 8
            outputs_2 = dense(W // 2, name='mlp2_{0}'.format(i))(outputs_2)

        rgb_nolight = dense(3, act=tf.keras.activations.relu)(outputs_2)
        rgb = rgb_nolight * dot_lights_noneg


        ###### Shadow Prediction ######
        shadow_params = dense(256, act=None, name='shadow_params')(outputs)

        inputs_3 = tf.concat(
            [shadow_params, inputs_lights], -1)  # concat lightdirs
        outputs_3 = inputs_3

        for i in range(2):
            # my original iteration of neural brdf used 4
            outputs_3 = dense(W // 2, name='mlp3_{0}'.format(i))(outputs_3)

        shadow = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='shadow', bias_initializer=tf.keras.initializers.TruncatedNormal(mean=5.0, stddev=0.05))(outputs_3)


        shadow_rgb = rgb * shadow

        outputs = tf.concat([shadow_rgb, alpha_out, n_norm, albedo, shadow], -1)

    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    return model


# Ray with light direction
# Ray helpers

def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def get_rays_relight(H, W, focal, c2w, lightdir):
    """Get ray origins, directions from a pinhole camera."""
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    light_d = tf.broadcast_to(lightdir, tf.shape(rays_d))

    return rays_o, rays_d, light_d


def get_rays_relight_np(H, W, focal, c2w, lightdir):
    """Get ray origins, directions from a pinhole camera."""

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    light_d = np.broadcast_to(lightdir, np.shape(rays_d))  # light_d should be like rays_o

    return rays_o, rays_d, light_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]  # ???
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * \
         (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * \
         (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds - 1)
    above = tf.minimum(cdf.shape[-1] - 1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape) - 2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape) - 2)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network_relight(inputs, viewdirs, lightdirs, fn, embed_fn, embeddirs_fn_views, embeddirs_fn_lights,
                        netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""

    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)

    if (viewdirs is not None) and (lightdirs is not None):
        input_dirs_view = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat_view = tf.reshape(input_dirs_view, [-1, input_dirs_view.shape[-1]])

        embedded = tf.concat([embedded, input_dirs_flat_view], -1)

        input_dirs_light = tf.broadcast_to(lightdirs[:, None], inputs.shape)
        input_dirs_flat_light = tf.reshape(input_dirs_light, [-1, input_dirs_light.shape[-1]])

        embedded = tf.concat([embedded, input_dirs_flat_light], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs_ = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])


    return outputs_


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

        rgb = raw[..., :3]

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        eps = 1e-05
        binary_loss = tf.math.log(alpha + eps) + tf.math.log(1 - alpha + eps)
        alpha_loss = alpha

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * \
                  tf.math.cumprod(1. - alpha + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(
            weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

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

        return rgb_map, disp_map, acc_map, weights, depth_map, alpha_loss, binary_loss

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

    # Points in space to evaluate model at. add noise to ray direction
    # print('ray_noise:',ray_noise)
    # pts = rays_o[..., None, :] + (rays_d[..., None, :]+ 0.001*tf.random.normal(rays_d[..., None, :].shape)) * \
    #       z_vals[..., :, None]  # [N_rays, N_samples, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
                z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # Evaluate model at each point.
    # raw = network_query_fn(pts, viewdirs, network_fn)  # [N_rays, N_samples, 4]
    raw = network_query_fn_relight(pts, viewdirs, lightdirs, network_fn)  # [N_rays, N_samples, 4]

    rgb_map, disp_map, acc_map, weights, depth_map, alpha_loss, binary_loss = raw2outputs(
        raw, z_vals, rays_d)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # Obtain additional integration times to evaluate based on the weights
        # assigned to colors in the coarse model.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
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
        rgb_map, disp_map, acc_map, weights, depth_map, alpha_loss, binary_loss = raw2outputs(
            raw, z_vals, rays_d)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['alpha_loss'] = alpha_loss
        ret['binary_loss'] = binary_loss
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
                   use_viewdirs=False, use_lightdirs=False, c2w_staticcam=None, ray_noise=1.0,
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
        rays_o, rays_d, light_d = get_rays_relight(H, W, focal, c2w, lightdir)  # kinda fixed
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
        rays_o, rays_d = ndc_rays(
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


def render_relight_path(render_poses, lightdir, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0,
                        **kwargs):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        t = time.time()

        rgb, disp, acc, _ = render_relight(
            H, W, focal, lightdir, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)

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


def render_relight_path_multilight(render_poses, lightdirs, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None,
                                   render_factor=0, **kwargs):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        t = time.time()

        rgb, disp, acc, _ = render_relight(
            H, W, focal, lightdirs[i], chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)

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


def create_nerf_relight(args):
    """Instantiate NeRF's MLP model."""

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 3
    input_ch_lights = 3
    embeddirs_fn_views = None
    embeddirs_fn_lights = None

    output_ch = 4
    skips = [4]
    model = init_nerf_relight_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, input_ch_lights=input_ch_lights, use_viewdirs=args.use_viewdirs,
        use_lightdirs=args.use_lightdirs)
    print(model)
    # tf.keras.utils.plot_model(model, "relight_nerf_arch.png", show_shapes=True)  # uncomment to get a .png image for the model
    grad_vars = model.trainable_variables
    models = {'model': model}

    model_fine = None
    if args.N_importance > 0:
        model_fine = init_nerf_relight_model(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, input_ch_lights=input_ch_lights, use_viewdirs=args.use_viewdirs,
            use_lightdirs=args.use_lightdirs)
        grad_vars += model_fine.trainable_variables
        models['model_fine'] = model_fine

    def network_query_fn_relight(inputs, viewdirs, lightdirs, network_fn):
        return run_network_relight(
            inputs, viewdirs, lightdirs, network_fn,
            embed_fn=embed_fn,
            embeddirs_fn_views=embeddirs_fn_views,
            embeddirs_fn_lights=embeddirs_fn_lights,
            netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn_relight': network_query_fn_relight,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'use_lightdirs': args.use_lightdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'ray_noise': args.ray_noise
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')
    parser.add_argument("--lightdirsdir", type=str,
                        default='./data/llff/fern', help='input data directory')
    parser.add_argument("--spherifydir", type=str,
                        default='./data/llff/fern', help='input data directory')                    

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')

    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--use_lightdirs", action='store_true',
                        help='use light direction input')
    parser.add_argument("--lightdir", type=int, default=[1.0, 0.0, 0.0],
                        help='one light direction input')
    parser.add_argument("--multilightdirs", action='store_true',
                        help='use multiple light directions for video rendering')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_output_type", type=str, default='rgb',
                        help='options: rgb / albedo / normal')
    parser.add_argument("--ray_noise", type=float, default=1.,
                        help='ray jitter')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_false',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
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
    parser.add_argument("--i_print", type=int, default=10,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=20,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    # loss options
    parser.add_argument("--sil_density_factor", type=float, default=0.01,
                        help='frequency of render_poses video saving')
    parser.add_argument("--binary_density_factor", type=float, default=0.0005,
                        help='frequency of render_poses video saving')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # Load data

    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, factor=args.factor,
                                                                  recenter=False, bd_factor=.75,
                                                                  spherify=args.spherify)

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
    print('shape', light_dirs.shape)

    c2w_spherify = np.load(args.spherifydir, )
    light_dirs = np.matmul(light_dirs, c2w_spherify[:,0:3])
    print('shape', light_dirs.shape)


    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf_relight(args)

    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching.
        #
        # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
        # interpreted as,
        #   axis=0: ray origin in world space
        #   axis=1: ray direction in world space
        #   axis=2: observed RGB color of pixel
        #   axis=3: light direction in world space
        print('get rays')
        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3], light_origin=[H, W, 3], light_direction=[H, W, 3]
        # for each pixel in the image. This stack() adds a new dimension.

        rays = [get_rays_relight_np(H, W, focal, p, l) for p, l in zip(poses[:, :3, :4], light_dirs)]
        rays = np.stack(rays, axis=0)  # [N, ro+rd+ld, H, W, 3]
        print('done, concats')
        # [N, ro+rd+ld+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
        # [N, H, W, ro+rd+ld+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+ld+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 4, 3])  # 1st 3 to 4 to include ld
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0

    N_iters = 1000005
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start, N_iters):
        time0 = time.time()

        # Sample random ray batch

        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2(ro,rd)+1(ld)+1(rgb), 3*?]
            batch = tf.transpose(batch, [1, 0, 2])  # [2(ro,rd)+1(ld)+1(rgb), B, 3*?]

            # batch_rays[i, n, xyz] = ray origin or direction or light direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:3], batch[
                3]  # change this from batch[:2] to batch[:3] to include light direction & batch[2] to batch[3] as rgb moved further

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)  # reshuffling again before epoch
                i_batch = 0

        #####  Core optimization loop  #####

        with tf.GradientTape() as tape:

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render_relight(
                H, W, focal, chunk=args.chunk, rays=batch_rays,
                verbose=i < 10, retraw=True, render_output_type=args.render_output_type, **render_kwargs_train)

            # Add gamma to rgb
            rgb_noneg = tf.nn.relu(rgb)
            gam = tf.convert_to_tensor(tf.constant([1 / 2.2]), dtype=tf.float32)
            rgb_noneg = tf.math.pow(rgb_noneg, gam)

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb_noneg, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                # print('img_loss0: ', img_loss0)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)

            # Add Silhoutte Loss
            if 'alpha_loss' in extras:
                alpha = extras['alpha_loss']
                mask = np.where((target_s[:,0].numpy()==0.0) & (target_s[:,1].numpy()==0.0) & (target_s[:,2].numpy()==0.0),0.0,1.0) # inverted black mask
                mask = np.reshape(mask, (1024,1))

                alpha = alpha*mask

                alpha_loss = tf.math.reduce_sum(alpha)/(args.N_rand*args.N_samples*2)
                weight_al = args.sil_density_factor  # 0.1 x 5 of loss
                w_alpha_loss = weight_al*alpha_loss
                loss += w_alpha_loss


            # Add Binary Loss
            if 'binary_loss' in extras:
                binary_loss = tf.math.reduce_sum(extras['binary_loss'])/(args.N_rand*args.N_samples*2)
                weight_al = args.binary_density_factor  # lowering it to zero from 0.0010
                w_binary_loss = weight_al*binary_loss
                loss += w_binary_loss



        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        dt = time.time() - time0

        #####           end            #####

        # Rest is logging

        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_video == 0 and i > 0:
            rgbs, disps = render_relight_path(
                render_poses, args.lightdir, hwf, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}_multilight'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            if args.multilightdirs:
                multilights = np.load('data/nerf_llff_data/readingPNG/multilights_z2.npy', ) #all DiLiGenT objects use same multi light setting

                # test view 1
                i_test_1 = i_test[1]
                render_relight_path_multilight_onepose(poses[i_test_1], multilights, hwf, args.chunk,
                                                       render_kwargs_test,
                                                       gt_imgs=images[i_test_1], savedir=testsavedir)

            else:
                render_relight_path(poses[i_test], args.lightdir, hwf, args.chunk, render_kwargs_test,
                                    gt_imgs=images[i_test], savedir=testsavedir)

            print('Saved test set')

        if i % args.i_print == 0 or i < 10:

            # print(expname, ' iter: ', i, ' psnr: ', psnr.numpy(), ' l_tot: ', loss.numpy(), ' l_imgloss: ', img_loss.numpy(), ' l_imgloss0: ', img_loss0.numpy(), ' w_silden_density: ', w_alpha_loss.numpy(), ' w_binary_density: ', w_binary_loss.numpy()) # uncomment if using Silhouette Density Loss
            print(expname, ' iter: ', i, ' psnr: ', psnr.numpy(), ' l_tot: ', loss.numpy(), ' l_imgloss: ', img_loss.numpy(), ' l_imgloss0: ', img_loss0.numpy(), ' w_binary_density: ', w_binary_loss.numpy()) # removed sil dens loss from printing
            print('iter time {:.05f}'.format(dt))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                # tf.contrib.summary.scalar('w_silden_loss', w_alpha_loss) # uncomment if using Silhouette Density Loss
                tf.contrib.summary.scalar('w_binary_loss', w_binary_loss)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)

            if i % args.i_img == 0:

                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3, :4]

                rgb, disp, acc, extras = render_relight(H, W, focal, args.lightdir, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                rgb = tf.math.pow(rgb, gam)

                psnr = mse2psnr(img2mse(rgb, target))

                # Save out the validation image for Tensorboard-free monitoring
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                if i == 0:
                    os.makedirs(testimgdir, exist_ok=True)
                imageio.imwrite(os.path.join(testimgdir, '{:06d}.png'.format(i)), to8b(rgb))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image(
                        'disp', disp[tf.newaxis, ..., tf.newaxis])
                    tf.contrib.summary.image(
                        'acc', acc[tf.newaxis, ..., tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])

                if args.N_importance > 0:
                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image(
                            'rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image(
                            'disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis])
                        tf.contrib.summary.image(
                            'z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis])

        global_step.assign_add(1)


if __name__ == '__main__':
    train()

