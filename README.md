Design space exploration using NSGA-II in MLPs for the Zynq-7000 device family . Includes a resource(FF,LUT,DSP) Zynq-7000 estimator and examples for both

## Dependencies
The dependencies and installs that are needed are:
dependencies:
  - python=3.10.10
  - jupyter_contrib_nbextensions==0.7.0
  - jupyterhub==3.1.1
  - jupyter-book==0.15.1
  - jsonschema-with-format-nongpl
  - pydot==1.4.2
  - graphviz==7.1.0
  - scikit-learn==1.2.2
  - tensorflow==2.11.1
  - tensorflow-datasets==4.8.3
  - webcolors
  - widgetsnbextension==3.6.0
  - pip==23.0.1
  - pip:
      - hls4ml[profiling]==0.8.0
      - qkeras==0.9.0
      - conifer==0.2b0
      - pysr==0.16.3

Vitis 2021 or newer is required to use hls4ml. The estimator was based on Vitis 2022.1

Alternatively the environment.yml configuration file can be used like this:

```console
conda create -f environment.yml 
conda activate DSE
```

##Outline
The code is organised into 3 main directories:
-MLP training for FF estimation(Training_MLPs)
-Estimator testing and LUT modeling(testing)
-DSE setup and examples



