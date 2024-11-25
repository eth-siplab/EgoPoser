# EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere (ECCV 2024, Official Code)

[Jiaxi Jiang](https://www.jiaxi-jiang.com/), [Paul Streli](https://www.paulstreli.com), [Manuel Meier](http://northwards.ch/), [Christian Holz](https://www.christianholz.net)<br/>

[Sensing, Interaction & Perception Lab](https://siplab.org), Department of Computer Science, ETH Zürich, Switzerland <br/>

[![project page](https://img.shields.io/badge/project_page-siplab_ethz-blue.svg)](https://siplab.org/projects/EgoPoser)
[![paper download](https://img.shields.io/badge/arxiv-paper-red.svg)](https://arxiv.org/abs/2308.06493)

<p align="center">
</p>

___________
![teaser](figs/teaser.jpg)
> Full-body egocentric pose estimation from head and hand poses alone has become an active area of research to power articulate avatar representations on headset-based platforms. However, existing methods over-rely on the indoor motion-capture spaces in which datasets were recorded, while simultaneously assuming continuous joint motion capture and uniform body dimensions. We propose EgoPoser to overcome these limitations with four main contributions. 1) EgoPoser robustly models body pose from intermittent hand position and orientation tracking only when inside a headset's field of view. 2) We rethink input representations for headset-based ego-pose estimation and introduce a novel global motion decomposition method that predicts full-body pose independent of global positions. 3) We enhance pose estimation by capturing longer motion time series through an efficient SlowFast module design that maintains computational efficiency. 4) EgoPoser generalizes across various body shapes for different users. We experimentally evaluate our method and show that it outperforms state-of-the-art methods both qualitatively and quantitatively while maintaining a high inference speed of over 600fps. EgoPoser establishes a robust baseline for future work where full-body pose estimation no longer needs to rely on outside-in capture and can scale to large-scale and unseen environments.

Method Overview
----------
![architecture](figs/architecture.jpg)



Our code is based on our previous project [AvatarPoser (ECCV'22)](https://github.com/eth-siplab/AvatarPoser) and is expected to be released in December.

Bibtex
----------

If you find our paper or codes useful, please cite our work:

	@inproceedings{jiang2024egoposer,
	  title={EgoPoser: Robust real-time egocentric pose estimation from sparse and intermittent observations everywhere},
	  author={Jiang, Jiaxi and Streli, Paul and Meier, Manuel and Holz, Christian},
	  booktitle={European Conference on Computer Vision},
	  year={2024},
	  organization={Springer}
	}	  

## License
This project is released under the MIT license. Our code is based on our previous project [AvatarPoser (ECCV'22)](https://github.com/eth-siplab/AvatarPoser).
