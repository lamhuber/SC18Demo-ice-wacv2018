## Multi-Task Spatiotemporal Neural Networks for Structured Surface Reconstruction

Created by Mingze Xu at Indiana University, Bloomington, IN

***Note:*** The released code and pretrained models are reimplemented on the following larger dataset.

### Environment

The code is developed with CUDA 8.0, ***Python 2.7***, ***PyTorch >= 0.4***

### Data Preparation

Download the raw data at ftp://data.cresis.ku.edu/data/rds/2014_Greenland_P3/CSARP_music3D/

Download the human-labled annotations at ./data/target.tar.gz

If you don't want to preprocess the data yourself, please use create_slices.m to generate radar images and convert_mat_to_npy.py to convert them from MATLAB to NumPy files.

And make sure to put the files as the following structure:
  ```
  data_root
  ├── slices_mat_64x64
  |   ├── 20140325_05
  │   ├── 20140325_06
  |   ├── 20140325_07
  │   ├── ...
  |
  ├── slices_npy_64x64
  |   ├── 20140325_05
  │   ├── 20140325_06
  |   ├── 20140325_07
  |   ├── ...
  |
  └── target
      ├── Data_20140325_05_001.txt
      ├── Data_20140325_05_002.txt
      ├── Data_20140325_06_001.txt
      ├── ...
  ```

### Pretrained Models

Download the pretrained model at ./pretrained_models

### Demo
To run the demo:
```
python demo.py --data_root {path/to/data_root} --c3d_pth {path/to/the/c3d.pth} --rnn_pth {path/to/the/c3d.pth}
```

### Citations

If you are using the data/code/model provided here in a publication, please cite our papers:

    @inproceedings{ice2018wacv, 
        title = {Multi-Task Spatiotemporal Neural Networks for Structured Surface Reconstruction},
        author = {Mingze Xu and Chenyou Fan and John Paden and Geoffrey Fox and David Crandall},
        booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
        year = {2018},
    }

    @inproceedings{icesurface2017icip, 
        title = {Automatic estimation of ice bottom surfaces from radar imagery},
        author = {Mingze Xu and David J. Crandall and Geoffrey C. Fox and John D. Paden},
        booktitle = {IEEE International Conference on Image Processing (ICIP)},
        year = {2017},
    }
