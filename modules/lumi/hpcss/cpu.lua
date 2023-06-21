-- CSC HPC Summmerschool module for CPU programming

help('CPU programming environment for CSC HPC Summerschool')

depends_on('LUMI/22.08')
depends_on('partition/C')
depends_on('PrgEnv-cray')
depends_on('cce/14.0.2')

prepend_path('PATH', '/project/project_465000536/bin')
