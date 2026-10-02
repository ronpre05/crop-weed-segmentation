# Crop and Weed Segmentation with Limited Data

Semantic segmentation of agricultural field images into **background**, **crop** and **weed**, trained on only 50 labelled images. Implemented in MATLAB with a U-Net.

The question behind the project: when labelled data is scarce, how much can be gained from training choices alone, without making the network any bigger? Both models here use the same depth-2 U-Net, so every difference in the results comes from augmentation and optimisation.

## Results

Evaluated on the same held-out split of 10 images (40 used for training).

| Model | Global accuracy | Mean accuracy | Mean IoU | Weighted IoU | Mean BF score |
|---|---|---|---|---|---|
| Baseline | 0.9700 | 0.7076 | 0.6064 | 0.9521 | 0.7156 |
| Improved | 0.9761 | 0.7617 | **0.6845** | 0.9578 | 0.7513 |

Per-class IoU:

| Class | Baseline | Improved |
|---|---|---|
| Background | 0.9862 | 0.9854 |
| Crop | 0.3570 | 0.5150 |
| Weed | 0.4760 | 0.5533 |

Background dominates the images, so global accuracy is high for both models and says little. Mean IoU is the fairer measure, and the gain comes almost entirely from the two plant classes, with crop IoU rising by 16 points.

## What changed between the two models

| | Baseline (`segment_base.m`) | Improved (`segment_imp.m`) |
|---|---|---|
| Network | U-Net, encoder depth 2 | U-Net, encoder depth 2 |
| Input size | 360 x 480 x 3 | 360 x 480 x 3 |
| Augmentation | None | Horizontal reflection |
| Optimiser | SGDM | Adam |
| Learning rate | 1e-3 | 5e-4 |
| Mini-batch size | 2 | 1 |
| Epochs | 15 | 15 |

Things that were tried and did not help:

- **Stronger augmentation** (rotation and translation) produced unrealistic plants and unstable training on a dataset this small.
- **Class-weighted loss** to counter the background imbalance scored worse than the unweighted loss. The option is still in the script behind `useClassWeights`.

## Running it

Requires MATLAB with the Deep Learning Toolbox, Computer Vision Toolbox and Image Processing Toolbox.

The dataset is not included in this repository. The images come from the sugar beet field dataset of Chebrolu et al. (2017). Place the image/mask pairs next to the scripts:

```
cw_data/
  images/         field images
  segmentation/   masks with matching file names (black = background, green = crop, red = weed)
```

Then run `segment_base` or `segment_imp` in MATLAB. Each script checks that images and masks match, trains the network, saves it, and writes evaluation metrics. Set `runBaseline` / `runImproved` to `false` to skip training and evaluate the saved network instead, or `quickTestMode` to `true` for a short run.

The trained networks from the results above are included as `segmentnet_base.mat` and `segmentnet_imp.mat`.

## Limitations

Crop and weed remain much harder than background, particularly thin leaves and very small plants. With a validation set of 10 images the figures above are indicative rather than precise.

## References

- O. Ronneberger, P. Fischer, T. Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI, 2015.
- N. Chebrolu et al. "Agricultural robot dataset for plant classification, localization and mapping on sugar beet fields." The International Journal of Robotics Research, 36(10), 2017.

## Author

Ron Prekopuca
