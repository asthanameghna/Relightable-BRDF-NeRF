# Relightable-BRDF-NeRF

## [PAPER](https://arxiv.org/abs/2207.06793?context=cs) | [DATA](https://drive.google.com/drive/folders/15HSym1ra28T2ZAuDDj8PaoM3jPL-yn0e?usp=sharing) | [RESULTS](https://drive.google.com/drive/folders/1UmjR9IvIQVioeeB-9xMt9TwMOTfrBmOL?usp=sharing)

<img width="795" alt="Screenshot 2022-10-04 at 16 36 25" src="https://user-images.githubusercontent.com/34877328/193863236-8a1eac6a-db51-41a7-ba11-d0294072eb6f.png">



CONTENTS OF THIS PROJECT
---------------------
* Introduction
* Folder Structrue
* Requirements
* Installation
* Running & Training
* Results
* Maintainers & Collaborators
* Citation

INTRODUCTION
------------
We propose to tackle the multiview photometric stereo problem using an extension of Neural Radiance Fields (NeRFs), conditioned on light source direction. The geometric part of our neural representation predicts surface normal direction, allowing us to reason about local surface reflectance. The appearance part of our neural representation is decomposed into a neural bidirectional reflectance function (BRDF), learnt as part of the fitting process, and a shadow prediction network (conditioned on light source direction) allowing us to model the apparent BRDF. This balance of learnt components with inductive biases based on physical image formation models allows us to extrapolate far from the light source and viewer directions observed during training. We demonstrate our approach on a multiview photometric stereo benchmark and show that competitive performance can be obtained with the neural density representation of a NeRF.

<img width="782" alt="Screenshot 2022-10-04 at 17 36 25" src="https://user-images.githubusercontent.com/34877328/193876049-8ad04f56-1cdb-4665-9d06-245c16e0e545.png">


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

REQUIREMENTS
------------
This project requires the following modules:

 * [pip](https://pip.pypa.io/en/stable/installation/)
 * [numpy](https://pypi.org/project/numpy/)
 * [ConfigArgParse](https://pypi.org/project/ConfigArgParse/)
 * [imageio](https://pypi.org/project/imageio/)
 * [TensorFlow](https://www.tensorflow.org/install/pip)
 
 For exact versions, please refer to [requirements.txt](https://github.com/asthanameghna/Relightable-BRDF-NeRF/blob/main/requirements.txt).
 
INSTALLATION
------------
 
 * Download the entire project either using git clone or download as zip option.

 * Once the folder is on your local device, open a new terminal window check the version of python on you device using ```python -v``` and install the compatible pip version for it.
 
 * Now check version of pip using ```pip -version``` and navigate to your project folder
 
 * Install all the packages in requirements.txt using pip (links for each installation command provided above).
 
 * Check for correct installation in your desired folder using the following commands:
   ```$ python```
   and it will show the follwing message:
   ```
   Python 3.8.8 (default, Apr 13 2021, 12:59:45) 
   [Clang 10.0.0 ] :: Anaconda, Inc. on darwin
   Type "help", "copyright", "credits" or "license" for more information.
   >>>
    ```
   Use command
   ```
   import tensorflow
   ```
   If the respose does not provide any error you have successfully installed all the      dependencies.
   
   
RUNNING & TRAINING
------------

* For training any object, use the command:
  ```
  python relight_brdf_nerf.py --config config/config_bearPNG.txt
  ```
* For obtaining albedo, rgb, normals and shadow renderings, use the command:
   ```
  python get_normals.py      # make sure to use the correct model_xxxxx.npy for a particular object
  ```

https://user-images.githubusercontent.com/34877328/193870883-97d2ef5c-eb8c-4bdb-be90-1cf364900f51.mp4

https://user-images.githubusercontent.com/34877328/193871194-a97326cf-3eff-4bf3-a927-532c7f754fc3.mp4

## Generating poses for your own scenes

### Don't have poses?

We recommend using the `imgs2poses.py` script from the [LLFF code](https://github.com/fyusion/llff). Then you can pass the base scene directory into our code using `--datadir <myscene>` along with `-dataset_type llff`. You can take a look at the `config_bearPNG.txt` config file for example settings to use for a forward facing scene. For a spherically captured 360 scene, we recomment adding the `--no_ndc --spherify --lindisp` flags.

### Already have poses!

In `relight_brdf_nerf.py` and all other code, we use the same pose coordinate system as in OpenGL: the local camera coordinate system of an image is defined in a way that the X axis points to the right, the Y axis upwards, and the Z axis backwards as seen from the image.

Poses are stored as 3x4 numpy arrays that represent camera-to-world transformation matrices. The other data you will need is simple pinhole camera intrinsics (`hwf = [height, width, focal length]`) and near/far scene bounds. Take a look at [our data loading code](https://github.com/asthanameghna/Relightable-BRDF-NeRF/blob/e5a2c7c3d4a6c89b19302188c46cbf24da87c50d/relight_brdf_nerf.py#L835) to see more.

Our work has been inspired from the original [NeRF paper](https://arxiv.org/abs/2003.08934). You can check out their code [here](https://github.com/bmild/nerf).
  

MAINTAINERS & COLLAOBRATORS
-----------

Please reach out to us if you have any questions:
 * [Meghna Asthana](https://www.cs.york.ac.uk/people/asthana) (University of York) 
 * William Smith (University of York)
 * Patrik Huber (University of York)

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
