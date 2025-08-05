# derivatives_template_code

Scripts for preprocessing (longitudinal) fMRI data

## Installation

### Prerequisites

* A BIDS dataset in [DataLad dataset](http://docs.datalad.org/en/stable/generated/datalad.api.Dataset.html), which you can create using our [BIDS template](https://github.com/SkeideLab/bids_template) or download from [OpenNeuro](https://openneuro.org) (currently untested)

```diff
+ templateflow is needed, but not in the environment yet: pip install templateflow (conda not available) @@

+ Get a freesurfer license (for Linux!) here: https://surfer.nmr.mgh.harvard.edu/registration.html @@
+ License will be mailed to you in 5-15 minutes.
```

### If you already have a `derivatives` sub-dataset

* If using our BIDS template, your BIDS dataset probably already has a sub-dataset called `derivatives` installed
* You can install the scripts from this repository using the following command from your main BIDS directory:

    ```bash
    datalad install -d . -s https://github.com/SkeideLab/derivatives_template_code.git derivatives/code
    ```

### If you don't have a `derivatives` sub-dataset

* If you don't have a `derivatives` sub-dataset installed in your BIDS dataset, you can do so using the following command from your main BIDS directory:

    ```bash
    datalad create -d . -c text2git derivatives
    ```

* Next, you can install the scripts from this repository using the following command:

    ```bash
    datalad install -d . -s https://github.com/SkeideLab/derivatives_template_code.git derivatives/code
    ```

```diff
+ ### Prepare the run_params.json

+ add the path to your emailed freesurfer license file
+ maybe change the default parameters, if you think it better fits your needs.
```

## Usage

```diff
+ python3 derivatives/code/run.py 
```

## Processing details

There are several different approaches to choose from, each its own branch. See each branch for its own processing details. 
- ``1-fmriprep``
- ``1-fmriprep-suma``
- ``2-freesurfer-fmriprep``
- ``3-freesurfer-long-fmriprep``
- ``4-freesurfer-split-fmriprep``
