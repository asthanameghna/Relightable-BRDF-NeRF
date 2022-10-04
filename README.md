# Relightable-BRDF-NeRF

## [PAPER](https://arxiv.org/abs/2207.06793?context=cs) | [DATA](https://drive.google.com/drive/folders/15HSym1ra28T2ZAuDDj8PaoM3jPL-yn0e?usp=sharing) | [RESULTS](https://drive.google.com/drive/folders/1t1e1a_a7Rh2lOOAowqqJeTlBcKWK_6kZ?usp=sharing)

<img width="795" alt="Screenshot 2022-10-04 at 16 36 25" src="https://user-images.githubusercontent.com/34877328/193863236-8a1eac6a-db51-41a7-ba11-d0294072eb6f.png">



CONTENTS OF THIS PROJECT
---------------------
* Introduction
* Folder Structrue
* Requirements
* Installation on Local Device
* Running on Local Device
* Results
* Maintainers
* Citation

INTRODUCTION
------------
We propose to tackle the multiview photometric stereo problem using an extension of Neural Radiance Fields (NeRFs), conditioned on light source direction. The geometric part of our neural representation predicts surface normal direction, allowing us to reason about local surface reflectance. The appearance part of our neural representation is decomposed into a neural bidirectional reflectance function (BRDF), learnt as part of the fitting process, and a shadow prediction network (conditioned on light source direction) allowing us to model the apparent BRDF. This balance of learnt components with inductive biases based on physical image formation models allows us to extrapolate far from the light source and viewer directions observed during training. We demonstrate our approach on a multiview photometric stereo benchmark and show that competitive performance can be obtained with the neural density representation of a NeRF.


FOLDER STRUCTURE
------------
```
.
├── c2w_matricies                         # camera to world matricies for DiLiGenT objects              
│   ├── spherify_c2w_bearPNG.npy                             
│   ├── spherify_c2w_buddhaPNG.npy                   
│   ├── spherify_c2w_cowPNG.npy 
│   ├── spherify_c2w_pot2PNG.npy                   
│   └── spherify_c2w_readingPNG.npy 
├── config                                # training configuration example for for DiLiGenT objects              
│   ├── config_bearPNG.txt                             
│   ├── config_buddhaPNG.txt                   
│   ├── config_cowPNG.txt
│   ├── config_pot2PNG.txt                   
│   └── config_readingPNG.txt
├── data                                 # place downloaded data here        
│   └── data_info.txt                    # link to G-Drive file for downloading the data
├── logs                                 # place downloaded logs here         
│   └── logs_info.txt                    # link to G-Drive file for downloading experiment logs
|
├── LICENSE.md
├── README.md
|
├── run.sh                               # bash file for command line run
|
├── get_albedos.py                       # Render albedos for trained model
├── get_normals.py                       # Render normals for trained model
├── get_rgbs.py                          # Render RGBs for trained model
├── get_shadows.py                       # Render shadows for trained model
|
├── load_llff.py                         # helper script to creat world to nerf (w2n) matricies
├── relight_brdf_nerf.py                 # relight-BRDF-nerf model + runner
|
├── requirements.txt                     # prerequisites for run
└── top_view_renderer.py                 # Render top head views for trained model

```

CITATION
------------
```
@article{asthana2022neural,
  title={Neural apparent BRDF fields for multiview photometric stereo},
  author={Asthana, Meghna and Smith, William AP and Huber, Patrik},
  journal={arXiv preprint arXiv:2207.06793},
  year={2022}
}
```
