# Caffe - Programming tools

## Content

- `metrics`: Particular metrics and related functions to derive predictions from a blob/net/HDF5 inference file.
- `models`
- `plot`: Automatic plot from Caffe logs.
- `train`: Pythonic approach of Caffe training. Nets are trained using the python interface which lets the user control each training step and perform tasks between iterations (e.g. compute any wanted metrics without nedding extra-layers in Caffe). Usual GLOG logs are replaced by custom logs by capturing the STDERR stream.
- `utils`: Collection of tools to help with logging, stream grabbing, etc.
