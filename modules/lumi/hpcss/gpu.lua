-- CSC HPC Summmerschool module for GPU programming

help('GPU programming environment for CSC HPC Summerschool')

depends_on('LUMI/22.08')
depends_on('partition/G')
depends_on('PrgEnv-cray')
depends_on('cce/14.0.2')
depends_on('rocm/5.3.3')

prepend_path('PATH', '/project/project_465000536/bin')
