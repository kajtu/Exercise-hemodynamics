# Exercise hemodynamics multi-timescale model
This is the code and data for the manuscript "Unraveling patient-specific responses to exercise in hypertension using exercise MRI and cardiovascular digital twins" by  Kajsa Tunedal, Henrik Ekman, Valentin Kindesjö, Magnus Wijkman, Karin Rådholm, Carl-Johan Carlhäll, Gunnar Cedersund, Tino Ebbers. This work was first published as a manuscript in the PhD thesis "Unraveling Patient-Specific Mechanisms of Hypertension Using Mathematical Modeling and MRI" by Kajsa Tunedal (DOI: [10.3384/9789181183771](https://doi.org/10.3384/9789181183771)).

If you are using any part of this code in a scientific context, please cite this paper.


## Requirements
This project was written on Windows 10 using python 3.13, and has been tested on Windows 11 and CentOS 7.
The models are run using the [SUND toolbox](https://pypi.org/project/sund/). SUND verison 1.6.4 was used during model development, parameter estimation, and results creation, but the results are here replicated with the latest version as of 20260630 - version 2.4.1.

All required packages are listed in `requirements.txt` and in the `pyproject.toml` file.

uv was used to run the code with a virtual environment. If you are using uv, set up the project by running `uv sync` in the terminal.

## Code structure
The model simulation, parameter estimation, and plotting are performed by the main script `main.py` using three input arguments (model name, number of optimizations, and a flag for running uncertainty estimation). Plotting data is done by `plot_mechamp_data.py`. Helper functions are found in the folder `functions`.

## Re-create manuscript figures
To plot the results from the manuscript, run:

1. `python plot_mechamp_data.py` for the data only (Figure 3)
2. `python main.py 'bp_exercise_fb_both'` for the simulations and literature data (Figure 4)
3. `python main.py 'bp_exercise_fb_both_rand'` for the simulations and subject-specific data for the full exercise session (Figure 5 and 8, Table 6)
4. `python main.py 'avatar_exercise'` for the simulations and subject-specific heartbeat data during rest and exercise (Figure 6 and 7, Table 7)

The resulting figures will be saved in the folder `figures` as .png and .svg, and tables will be saved in `tables` as excel files. Note that the final manuscript figures are created by combining or selecting elements from these figures to improve clarity and readability.

## Data
The literature data and the anonymized collected data used for model training and validation are available in the folder `data`. 

Literature data: `exercisedata_supine.json`. Note that the literature data is digitized data from previous studies, and errors might have occurred during curation of the data. The doi for each literature study is given in the json structure for each dataset. 

Anonymized collected data: `MeChaMPdata_anonymized.json`. Note that the numbers of the controls and subjects with hypertension do not correspond to the age- and sex- matched subjects (ie c1 is not the paired control to p1) in this anonymized dataset, and that paired testing cannot be performed.

## Models
The model files are found in the folder `models`. The model files are ODE models in the SUND toolbox format (https://pypi.org/project/sund/, documentation: https://isbgroup.eu/sund-toolbox/latest/).

There are three model files:
* `bp_exercise_fb_both.txt` is the base model for describing average data response to supine physical activity.
* `bp_exercise_fb_both_rand.txt` is the same model as `bp_exercise_fb_both.txt`, but with the addition of subject-specific random effects to describe subject-specific variation.
* `avatar_exercise_legs6.txt` is the cardiovascular lumped parameter model of the left heart and whole-body circulation.

### Estimated model parameters
The best parameters after estimation with data are saved in the folder `parameters`, with a separate folder for each model and a subfolder for each dataset/subject. In the parameters folders there is also a saved simulation of the exercise model for the relevant features used as input to the cardiovascular model at the wanted time points.

## Parameter estimation
The main script accepts `python main.py MODELNAME [nopts] [DOPPL]`. Valid model names are bp_exercise_fb_both, bp_exercise_fb_both_rand, and avatar_exercise. Omitting `nopts` or setting `nopts` to 0 means that no new parameter optimization is done, while `nopts`>0 starts a parameter optimization which is repeated `nopts` times. Setting the third argument to `DOPPL` triggers an uncertainty estimation for the given model.

Note that parameter estimation is computationally expensive. The parameter estimation was done by running the main script for the different models with the second argument nopts set > 0 (for example `uv run main.py 'bp_exercise_fb_both' 10` for the model bp_exercise_fb_both with 10 optimizations). This was run in parallel at the National Supercomputer Centre at Linköping University to reduce the computational time. During parameter estimation, results are saved in the folder `parameters` under the subfolder with the model name. 

Note that uncertainty estimation is computationally expensive. Model uncertainty was performed with the main script with the third argument DOPPL set to True and nopts>0 (for example `uv run main.py 'bp_exercise_fb_both' 10 DOPPL` for the model bp_exercise_fb_both with 10 optimizations) and were run in parallel at the National Supercomputer Centre at Linköping University. It is recommended to run the estimation on a similar HPC. The results are saved in `results_PPL` but since there is such a large number of files saved (at least two files per dataset/subject, timepoint and feature), these results are condensed into `parameters/MODELNAME/UC/` for each modelfile, where only the maximum and minimum found bounds for each variable is saved. Re-running the uncertainty estimation might overwrite the results.

## Licence
MIT License

Copyright (c) 2026 Kajsa Tunedal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.