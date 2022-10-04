
import os, sys
from platform import architecture
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import numpy as np
import imageio
import pprint

import relight_brdf_nerf

##################### This file renders top-head view for any trained model ###################

def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)


basedir = './logs'
##### Multiple pre-trained folders ######
# expname = 'exp267_cowPNG'
# expname = 'exp251_buddhaPNG'
# expname = 'exp205_readingPNG'
# expname = 'exp206_pot2PNG'
expname = 'exp204_bearPNG' #your path to trained log folder
architecture = relight_brdf_nerf
model_no = 400000

config = os.path.join(basedir, expname, 'config.txt')
print('Args:')
print(open(config, 'r').read())

parser = architecture.config_parser()
ft_str = ''
ft_str = '--ft_path {}'.format(os.path.join(basedir, expname, 'model_{:06d}.npy'.format(model_no)))
args = parser.parse_args('--config {} '.format(config) + ft_str)

# Create nerf model
render_kwargs_train, render_kwargs_test, start, grad_vars, models = architecture.create_nerf_relight(args)

bds_dict = {
    'near' : tf.cast(2., tf.float32),
    'far' : tf.cast(6., tf.float32),
}
render_kwargs_test.update(bds_dict)

print('Render kwargs:')
pprint.pprint(render_kwargs_test)

net_fn = render_kwargs_test['network_query_fn_relight']
# net_fn = render_kwargs_test['network_query_fn']
print(net_fn)

# Render an overhead view to check model was loaded correctly
c2w = np.eye(4)[:3,:4].astype(np.float32) # identity pose matrix
c2w[2,-1] = 4.

H, W, focal = 257, 307, 300 #DiLiGenT experimental to fix 2D plot

testimgdir = os.path.join(basedir, expname, 'top_views')
os.makedirs(testimgdir, exist_ok=True)

down = 1
lightdirs = np.load('data/nerf_llff_data/readingPNG/multilights_z2.npy',) #all DiLiGenT objects use same multi light setting
lightdir = []
for i in range(0,len(lightdirs)):
    lightdir = lightdirs[i]
    test = architecture.render_relight(H//down, W//down, focal/down, lightdir, c2w=c2w,  **render_kwargs_test)
    img = 2*(np.clip(test[0],0,1))
    rgb8 = to8b(img)
    filename =  'logs/exp204_bearPNG/top_views/v{:03d}.png'.format(i)
    imageio.imwrite(filename, rgb8)