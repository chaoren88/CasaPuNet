# CasaPuNet: Channel Affine Self-Attention Based Progressively Updated Network for Real Image Denoising

## Environment

The model is built in PyTorch 1.7.1 and tested on Ubuntu 20.04 environment (Python3.7, CUDA10.2).

## Download

Pretrained Model: https://drive.google.com/file/d/1lTojt_U10Lj6IzgvrlXRk6p7gMkHDEIQ/view?usp=share_link

DND Dataset: https://noise.visinf.tu-darmstadt.de/downloads/

SIDD Dataset: https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php

CT Core Image Dataset: https://drive.google.com/file/d/1QWPj0OMfbgNT4cgucFjL3YTqmpWNc5c4/view?usp=share_link

## Test

Extract the files to `dataset` folder and `checkpoint` folder as follow:

```
~/
  dataset/
    benchmark/
      dnd_2017/
        images_srgb/
            ... (mat files)
            ... (mat files)
        info.mat
      sidd/
        BenchmarkNoisyBlocksSrgb.mat
  checkpoint/
    checkpoint.pth.tar
```

To test on DND or SIDD Benchmark, run

```
python test_benchmark.py --type dnd_or_sidd
```

To test on noisy images, run

```
python test_image.py
```
