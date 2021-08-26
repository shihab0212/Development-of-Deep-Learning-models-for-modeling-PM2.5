<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

------------------------------------

### Built With

This section should list any major frameworks that you built your project using:
* conda >=4.7.12

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running install conda and follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.

### Installation

1. Clone the repo: https://github.com/rashikcs/Deep_learning_for_air_quality_prediction.git

2. - Install required packages packages by going to the project folder and type this from command prompt
   ```sh
   conda env update -f environments.yml 
   ```
   - If you have Cuda compatible NVIDIA GPU proceed with this
   ```sh
   conda env update -f environments_gpu.yml 
   ```
This will create a new conda environment named **air_quality_environment**.<br />
Activate this environment from shell by typing
   ```sh
   conda activate air_quality_environment.yml
   ```
Run Jupyter Notebook after activating the environment.
   ```sh
   jupyter notebook
   ```
<!-- USAGE EXAMPLES -->
## Usage
1. For Prophet: Set max_number_of_trials 50 if low computational power in init_params.json.

  - Prepare and Train Model with optimization: Run all cells sequentially in Prophet_Train_HPO.ipynb file.
    - Best model will be saved "{dataset_name}\final_models\" folder.
    - Visualizations hasn't been saved for prophet
    
  - Forecasted results will be saved "visalizations\{dataset_name}\" folder.

2. For AutoEncoders: Three autoencoder models have been implemented. Run one by one changing params in init_params.json

  - Data Preparation: Run all cells sequentially in AE_Data_Preparation.ipynb file.
    -Prepared data will be saved "{dataset_name}/cross_validation/" folder.
    
  - Train & optimize model: Run all cells sequentially in AE_Train_model_HPO.ipynb file.
    - Trained models will be saved inside "{dataset_name}/final_models/"
    
  - Forecasted results and visualizations will be saved "{dataset_name}/visalizations\{model_name}\" folder.
  
3. For DeepAR: Set max_number_of_trials 50. With 2gb Nvidia GeForce GTX 1050 Ti it took 1 hour for 1 iteration of hyperparameter combination.

  - Data Preparation: Run all cells sequentially in DeepAR/DeepAR_data_preparation.ipynb file.
    - Prepared data will be saved "DeepAR\data\{dataset_name}" folder.
  - Train & optimize model: Run all cells sequentially in DeepAR/DeepAR_HPO_Train.ipynb file.
  - Forecasted results and visualizations will be saved "DeepAR\data\{dataset_name}\" folder.
  
In init_params.json parameters has been initialized. Check out the file and update necessary parameters before running any model.