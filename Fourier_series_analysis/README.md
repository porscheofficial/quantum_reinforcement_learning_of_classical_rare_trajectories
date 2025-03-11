# Quantum reinforcement learning of classical rare dynamics: analysis via Fourier series

This repository contains the code used in the publication "Quantum reinforcement learning (QRL) of classical rare dynamics:
Enhancement by intrinsic Fourier features"  (**TODO**: add DOI) to analyze its results via Fourier series.  
The code includes the following features (for more details see the publication):
* exact computation and evaluation of the reweighted (exponentially tilted) dynamics of 1D random walks 
* exact computation of the value function for the reweighted dynamics
* symbolic and numerical computation of the Fourier coefficients for the output of the parameterized quantum circuits 
  used to parameterize the policies of QRL agents
* fitting the parameterized policies (mainly in terms of their Fourier coefficients) to the reweighted dynamics 
* evaluation of the fitted policies
* plotting of all results


## Installation

Before running the script, make sure you installed Python 3.10 or higher and install the Python modules listed in 
"requirements.txt":

```bash
pip install -r requirements.txt
```


## Usage

In order to reproduce all results of this code including the plots used in the publication, run the script "main.py" 
in the terminal or console:

```
python3 main.py
```

If you wish, you can change the configuration parameters for the features listed above in the configfile 
"config_publication.json5" (for the meaning of the parameters see the comments therein).  
In order to keep track of your changes, it might be advisable to copy "config_publication.json5", rename it, 
change the parameters in the renamed configfile and change the keyword argument "param_file_name" in the 
function call "main()" in main.py accordingly.

When you run the script, the steps of the computations conducted and some details will be logged, by default in the file 
"script.log".  
The name of the logfile as well as the logging level can be changed in the file "logging_config.py" under the command 
"logging.basicConfig()".  
For the most time-consuming steps a progress bar is printed in the terminal (note that the estimation 
of the remaining time needed might not be accurate, especially when the computations are parallelized).

Each run of "main.py" with a differently named configfile (explained above) generates a folder of the same 
name in the folder "results" with all the data generated during the script run. The resulting folder contains the 
following:
* configfile (default: "config_publication.json5")
* "computations" folder: contains the results of all computations conducted, saved in the form of .npz files which 
                         contain the attributes of the respective class where the results are computed
* "plots" folder: contains all plots generated

When you interrupt the script, rerun it with the same configfile, and the configuration parameter 
"recompute_stored" == false in the configfile, by default all results of already conducted computations will be loaded 
via the function load_or_compute_obj(..., recompute=recompute_stored); if you want to recompute all results, set 
"recompute_stored" = true.  
If you want to recompute only some of the results, you have to manually set "recompute=True" in the respective function 
calls in "main.py".

The other Python files provide the classes and functions used in "main.py" and are named according to their 
respective functionality.

**Feature planned for the future:** If you want to test yourself the implemented non-standard algorithms most important for this code, run the following
commands in the terminal or console:

```
pytest test_Fourier_series_analysis_and_fits.py
```

```
pytest test_reweighted_dynamics.py
```

```
pytest test_utilities.py
```

```
pytest test_value_functions.py
```

## Author and acknowledgements

**Name**: David A. Reiss  
**ORCID**: [0000-0002-5455-4071](https://orcid.org/0000-0002-5455-4071)  
**Affiliation**: Dahlem Center for Complex Quantum Systems and Physics Department, Freie Universität Berlin, 
                 Arnimallee 14, 14195 Berlin, Germany  
**Email**: david.reiss@fu-berlin.de  
**GitHub**: [@D-A-Reiss](https://github.com/D-A-Reiss)  
**Year**: 2024

I want to thank Alissa Wilms and Clemens Wickboldt for stimulating discussions.
**TODO:** Add Eduard Itrich here?

## License

See [LICENSE](./LICENSE.md).
