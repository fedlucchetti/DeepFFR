# DeepFFR


## Install and run without docker container
```bash
git clone git@github.com:fedlucchetti/DeepFFR.git
# build application
python setup.py install
```

## Install and run inside docker container

```bash
# Build docker image
./build_docker
# enter environment
./run_docker
# build application
python setup.py install
```

## Autoencoder deployment

```bash
from deepffr import DeepFilter
df       = DeepFilter.DeepFilter()
filtered = df.apply_filter(efr_waveform)
```
