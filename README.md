# Master Thesis: Novel preconditioners for iterative methods in Vecchia-Laplace approximations

This repository contains the code and experiments for the Master's thesis on preconditioning strategies for iterative methods applied to the Vecchia-Laplace approximation, developed using the [GPBoost](https://github.com/fabsig/GPBoost) library.

## Repository Structure

### 📁 GPBoost

This folder replicates the structure of the official GPBoost library, but includes **only the modified files** necessary to implement and evaluate the proposed preconditioners.

#### Modified files:
- **`GPBoost/src/GPBoost/CG_utils.cpp`**  
  - Introduced support for the new `HLFPC` (High-Low Frequency Pivoted Cholesky) preconditioner in the `CGVecchiaLaplaceVec` and `CGTridiagVecchiaLaplace` routines.
  - Implemented several new methods for computing preconditioner factorizations:
    - Some variants of the `ReverseIncompleteCholeskyFactorization` (for `ZIRC`)
    - `PivotedCholeskyFactorizationForHLFPC`
    - `LanczosTridiagVecchiaLaplaceNoPreconditioner`

- **`GPBoost/include/GPBoost/re_model_template.h`**  
  - Added the new preconditioner types and configuration options to the model template class.

- **`GPBoost/include/GPBoost/CG_utils.h`**  
  - Declared the new functions implemented in `CG_utils.cpp`.

- **`GPBoost/include/GPBoost/likelihoods.h`**  
  - Integrated `HLFPC` method as a new preconditioner with all it's variants.

> 📌 Only the files relevant to the implementation of the new preconditioners are included to keep the codebase minimal. If you want to reproduce the experiments copy and paste the files in the same directory.

---

### 📁 Experiments

This folder contains two R scripts used to run experiments and visualize results.

#### 1. `ADA_Comparison_Preconditioner_Vecchia_x3.R`  
This script runs the experimental evaluations. It supports the following options:

- `TOY <- TRUE`  
  Runs the experiments on a smaller synthetic dataset with **10,000** points instead of the default **100,000**, useful for quicker tests and debugging.
  
- `LOAD <- TRUE`  
  Loads an existing dataset from `AllResults.rds`. Uncomment preconditioners to load from the `preconditioners_to_load` list.

- `SAVE <- TRUE`  
  Saves the experiment results to `AllResults.rds` at the end of execution.

To **run experiments**, uncomment the relevant preconditioners in the `PRECONDITIONER` list. Ensure the preconditioners to load are uncommented from both sets of preconditioners (`PRECONDITIONER` AND `preconditioners_to_load`) if `LOAD <- TRUE`.

#### 2. `Visualization_of_data.R`  
This script generates plots from the experimental results saved in `AllResults.rds`.

- `SDxTime <- TRUE`  
  Plots the standard deviation of the log-likelihood as a function of runtime.

- `SDxTime <- FALSE`  
  Two options available:
  - `variance <- TRUE`: plots the **variance** and runtime against the number of random vectors.
  - `variance <- FALSE`: plots the **difference to Cholesky** and runtime against the number of random vectors.

As in the first script, uncomment the desired preconditioners in the `PRECONDITIONER` list to select them for visualization.

### Reference and Adapted Code

The experiment code was developed by adapting the original simulation script from:  
📎 **[TimGyger/iterativeFSA - Comparison_Preconditioner_Vecchia.R](https://github.com/TimGyger/iterativeFSA/blob/main/Simulation_Studies/Comparison_Preconditioner_Vecchia.R)**

This served as the foundation for running and evaluating the newly proposed preconditioners in the context of the Vecchia-Laplace approximation.

---

### 🖥️ Hardware Specifications

All experiments were conducted on the **ADA-23** server provided by the Institute for Statistical and Mathematical Modeling at ETH Zurich. The server has the following hardware specifications:

- **CPU**: 2× Intel Xeon Gold 6148 (20 cores, 2.4 GHz)  
- **GPU**: 4× NVIDIA GeForce GTX 1080 Ti (12 GB each)  
- **RAM**: 256 GB

More information about the ADA cluster can be found at the official ETH Zurich page:  
🔗 [https://blogs.ethz.ch/isgdmath/central-clients/](https://blogs.ethz.ch/isgdmath/central-clients/)

---

## License

This project inherits the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0) from GPBoost.
