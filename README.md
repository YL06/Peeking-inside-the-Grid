# Peeking-inside-the-Grid
Experimental part of the Master Thesis "Peeking Inside the Grid: Membership Inference and Model Inversion Attacks on XAI-Based Smart Grid Systems" of Yejin Lee at Institute for Automation and Applied Informatics, Karlsruhe Institute of Technology.

The experiments addresses the privacy implications of applying XAI to ML models in smart grid systems. It examines two privacy attacks, Membership Inference ATtack and Model Inversion Attack, and investigates whether the availability of explanation information extends the attack surface. The goal is to assess whether privacy risks previously identified in image-based settings also arise for tabular data used in smart grid applications, thereby contributing to a better understanding of the trade-off between explainability and privacy in energy systems.

## Repositoy Structure
```
Peeking-inside-the-Grid/
├──eval  # evaluation of the results
├──model_output # results and target model
├──utils
├──MIA_RF_Smart Grid Stability.ipynb # Experiment: Membership Inference Attack utilizing XAI
├──MInvA_CNN_Smart Grid Real-Time Load Monitoring Dataset.ipynb # Experiment: Model Inversion Attack utilizing XAI
├──README.md

```

## Installation
1. Clone the repo
   ```sh
   git clone https://github.com/YL06/Peeking-inside-the-Grid.git
   cd Peeking-inside-the-Grid
   ```
2. Setup environment
   **With conda:**
	```
    conda create --name py311 python=3.11
    conda activate py311
	```

	```
	pip install -r requirements.txt
 	```
	**Or**
	```
 	pip install tensorflow==2.19.0
	pip install jupyterlab
	pip install jupyter
	pip install torch (no need?)
	pip install pandas
	pip install scikit-learn
	pip install kaggle
	pip install kagglehub
	pip install adversarial-robustness-toolbox
	pip install matplotlib
	pip install seaborn
	pip install shap
 	pip install pypickle
	```
4. Run JupyterLab
    ```
    jupyter lab
    ```

## Requirements
    python~=3.11
	tensorflow==2.19.0
	keras==3.14.1
	scikit-learn==1.9.0
