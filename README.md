# BANF: Band-limited Neural Fields for Levels of Detail Reconstruction

| [Project Page](https://theialab.github.io/banf/) | [Paper](https://arxiv.org/abs/2404.13024) |

This is the official repository for the paper "Band-limited Neural Fields for Levels of Detail Reconstruction" (CVPR 2024).

We present our results in several settings. You can find more details in the respective directories.
* [NeRFing](NeRFing/README.md)

<img src="NeRFing/vis_data/NeRFing.gif" width="550">


---

* [Multiview 3D reconstruction](3D_reconstruction/README.md)

<img src="3D_reconstruction/banf/teaser.png" width="550">

---

* [2D Image Fitting](2D_fitting/README.md)

<img src="2D_fitting/vis_data/2d_fitting.gif" width="550">


# Credits
This project is built on top of [SDFstudio](https://github.com/autonomousvision/sdfstudio) and [Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp). We thank the maintainers for their contribution to the community!

# BibTeX
If you find BANF useful, please consider citing:
```
@article{ahan2023banf,
  author    = {Shabanov, Ahan and Govindarajan, Shrisudhan and Reading, Cody and Goli, Lily and Rebain, Daniel and Moo Yi, Kwang and Tagliasacchi, Andrea},
  title     = {BANF: Band-limited Neural Fields for Levels of Detail Reconstruction},
  year      = {2024},
  booktitle   = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```