-- CSC HPC Summmerschool module for GPU programming

help('GPU programming environment for CSC HPC Summerschool')

load('LUMI/22.08')
load('partition/G')
load('PrgEnv-cray')
load('cce/14.0.2')
load('rocm/5.3.3')

prepend_path('PATH', '/project/project_465000536/bin')
