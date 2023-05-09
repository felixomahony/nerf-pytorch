# Joint Training for Style Transfer
[PDF Report](report/project_report.pdf)

This git repository, modified from the original, is the code used in the final paper of Felix O'Mahony for class COS 526 - Joint Training for Style Transfer in NeRF.

The new method permits the modification of the style of a three-dimensional scene generated with NeRF. Whereas existing methods for style transfer typically rely on transforming the style of a NeRF model's training image set to adapt styles, we propose a method which modifies the model directly. In our method, a reference scene is generated in two styles. In one, the style resembles that of the target scene. Another serves as a desired style, which resembles the desired appearance of the target scene. By training two networks in parallel, we are able to transfer the style from the reference model in the desired style to the target scene.

![Example relighting](imgs/relight.gif)
*Example of a Relit Scene*

## Running the Scripts

To run the scrpits, first a config file must be established. Note that these are different to configs used for regular NeRF training, as they must specify *three* training data directories.

Some example data is given in `data` and an example config is given in `configs/spheres.txt`.

The script can be run locally from `run_compound_nerf.py`. This will train a model according to the given config file.

## Modifications to Original

Several files are modified and added to this project to make the script work.
- `./data/` Four new datasets are generated and placed here. `sphere/` is the reference scene in original style. `sphere_blue/`, `sphere_relit/`, `sphere_texture/` show the same content in different styles. Note that the actual images are not included as the `.gitignore` ignores image files, but if you would like to see them please message me.
- `./blender/` carries all the blender files used to produce the new scenes.
- `run_nerf_helpers` has been modified to split the network in two. The `FMap` class is the encoder, while `Appendix` is the decoder.
- `run_compound_nerf` has been modified to train across the three different combinations of encoder and decoder available. The training is, however, carried out in the same manner as it is originally.
- `run_compound_locally.py` is a simple testing script which can be used to test the method locally. However it is advised to use a GPU, so an HPC was actually used to produce the results shown in the paper.