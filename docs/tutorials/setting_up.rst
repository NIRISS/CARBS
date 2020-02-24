Installation
============

Clone and install the repository with:

    git clone https://github.com/NIRISS/CARBS.git
    cd CARBS
    python setup.py install
    
# Dependencies

- [dense_basis](https://dense-basis.readthedocs.io/en/latest/)
- [grizli](https://grizli.readthedocs.io/en/master/)
- [emcee](https://emcee.readthedocs.io/en/stable/)

## dense_basis dependencies:


- [FSPS](https://github.com/cconroy20/fsps) and [python-fsps](http://dfm.io/python-fsps/current/installation/) - needs .bashrc environmental variable. 
- astropy, scikit-learn and george. See the full list [here](https://dense-basis.readthedocs.io/en/latest/usage/dependencies.html).

## grizli dependencies:

- [installation instructions](https://grizli.readthedocs.io/en/master/grizli/install.html) / astroconda
- psycopg2
- astropy>=3.0
- scikit-image>=0.13.0
- scikit-learn>=0.19.0
- yaml>=0.1.7
- pyyaml>=3.12
- nose>=1.3.7
- cython>=0.28.2
- astroquery>=0.3.7
- photutils>=0.4
- pyregion>=2.0
- crds>=7.1.5
- drizzlepac>=2.2.2
- hstcal>=2.0.0
- pysynphot>=0.9.12
- stwcs>=1.4.0
- wfc3tools>=1.3.4
- shapely>=1.6.4
- descartes>=1.0.2
- boto3>=1.7.51
- peakutils>=1.0.3
- extinction>=0.3.0
- specutils==0.2.2
- git+https://github.com/gbrammer/sep.git
- git+https://github.com/gbrammer/pyia.git
- git+https://github.com/gbrammer/reprocess_wfc3.git
- git+https://github.com/gbrammer/eazy-py.git
- git+https://github.com/gbrammer/tristars.git
- git+https://github.com/gbrammer/mastquery.git


Put these lines in ~/.bashrc:

    export SPS_HOME = "${HOME}/fsps/src" # or anywhere else
    export GRIZLI="${HOME}/grizli" # or anywhere else
    export iref="${GRIZLI}/iref/"  # for WFC3 calibration files
    export jref="${GRIZLI}/jref/"  # for ACS calibration files

    # Make the directories, assuming they don't already exist
    mkdir $GRIZLI
    mkdir $GRIZLI/CONF      # needed for grism configuration files
    mkdir $GRIZLI/templates # for redshift fits

    mkdir $iref
    mkdir $jref