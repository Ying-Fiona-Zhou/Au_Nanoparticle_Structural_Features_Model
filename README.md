# Au Nanoparticle Growth Prediction Model

## Overview
This project provides a machine learning-based model to predict the **growth of gold nanoparticles (Au NPs)** under various synthesis conditions. The model leverages **nanoparticle structural features** such as **number of surface atoms (N_surface)**, **surface area-to-volume ratio (SA_to_V_ratio)**, **average radius (R_avg)**, and synthesis conditions like **temperature** and **time** to estimate the final size and surface characteristics of Au NPs.

The model is designed to be **scalable for industrial use** and can be integrated into **real-time synthesis** pipelines to optimize **nanoparticle production** for applications like **catalysis**, **drug delivery**, **environmental remediation**, and **solar energy**.

## Key Features
- Predicts **nanoparticle radius** and **surface properties** based on synthesis conditions.
- Implements various **growth models**, including:
  - **Diffusion-Limited Growth (DLG)**
  - **Aggregation Growth Model**
  - **Ostwald Ripening Model**
- Machine learning models such as **Random Forest Regression** and **Gradient Boosting** are used to improve predictions.
- Optimized using **GridSearchCV** and **RandomizedSearchCV** for hyperparameter tuning.
- **Feature importance analysis** is included to highlight the most impactful factors on nanoparticle growth.
  
## Applications
- **Pharmaceuticals**: Optimize **drug delivery nanoparticles** for targeted therapies.
- **Catalysis**: Enhance **nanoparticle surface area** for improved catalytic activity.
- **Environmental Remediation**: Design nanoparticles for **water filtration** and **pollution control**.
- **Solar Energy**: Improve the efficiency of **solar cells** by predicting the size of gold nanoparticles for optimal **light absorption**.
- **Electronics**: Develop consistent nanoparticle sizes for **semiconductors** and **nanoelectronics**.

## File Structure
- **Au_NP_Model.ipynb**: The main Jupyter Notebook containing the full analysis, model training, and evaluation steps.
- **Au_nanoparticle_dataset.csv**: The dataset used for training and testing the model, containing features such as temperature, time, radius, and surface properties of nanoparticles.
- **README.md**: This README file providing an overview of the project.
- **results/**: Folder containing any output plots or evaluation results from the model runs.

## Model Overview
The model workflow consists of the following key steps:
1. **Data Loading**: Load the dataset containing structural features of nanoparticles and synthesis conditions.
2. **Exploratory Data Analysis (EDA)**: Visualize key trends, such as nanoparticle radius over time or under different temperatures.
3. **Growth Models**:
   - **Diffusion-Limited Growth**: Predicts radius as a function of time using a diffusion-based model.
   - **Aggregation Model**: Predicts growth due to particle aggregation.
   - **Ostwald Ripening**: Predicts growth as smaller particles dissolve and redeposit onto larger ones.
4. **Machine Learning Models**:
   - **Random Forest Regression**: A tree-based ensemble model used for predicting nanoparticle size.
   - **Gradient Boosting Regressor**: An advanced boosting model used to improve prediction accuracy.
5. **Hyperparameter Tuning**:
   - Optimized the Random Forest and Gradient Boosting models using **GridSearchCV** and **RandomizedSearchCV** to find the best set of hyperparameters.
6. **Evaluation**: The model performance is evaluated using:
   - **R² (coefficient of determination)**: Measures how well the model fits the data.
   - **RMSE (Root Mean Squared Error)**: Measures the error between predicted and actual values.

## Results
- **Best Model**: The optimized Gradient Boosting Regressor achieved an R² of 0.997995 and an RMSE of 0.348732, indicating high accuracy in predicting nanoparticle radius.
- **Feature Importance**:
    - **N_surface (Number of Surface Atoms)**: The most important feature in determining nanoparticle growth.
    - **SA_to_V_ratio (Surface Area-to-Volume Ratio)**: Contributed significantly to the model’s predictions, particularly in growth-related properties.

## Visualization

Several plots are included in the notebook to help visualize model performance and key trends in the data:
- **Actual vs Predicted Nanoparticle Radius**: Compares model predictions to actual nanoparticle sizes.
- **Correlation Matrix**: Shows the relationships between different features in the dataset.
- **Feature Importance Plot**: Displays which features had the most impact on nanoparticle size prediction.

## Requirements
Before running the model, ensure the following libraries are installed:

- **Python 3.x**
- **NumPy**
- **Pandas**
- **Matplotlib** (for plotting)
- **Seaborn** (for correlation matrix and feature visualization)
- **Scikit-learn** (for machine learning models)
- **Jupyter Notebook** (for running the code)

To install the required libraries, you can run:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn

## Credits & Learning

#### Credits
- This project was inspired by research on **nanoparticle growth models** and the use of **machine learning in materials science**. 
- Special thanks to the developers of **Scikit-learn**, **Pandas**, and **Matplotlib**, whose tools were essential in building this project.
- Notable research papers that inspired this project:
  - *Diffusion-limited aggregation, a kinetic critical phenomenon* by Witten & Sander (1981)
  - *Ostwald Ripening: A Simplified Kinetic Model* by Lifshitz & Slyozov (1961)
  - *Machine learning for molecular and materials science* by Butler et al. (2018)

#### Learning Outcomes
Through this project, I gained significant insights into:
- **Nanoparticle Growth Mechanisms**: Understanding how physical processes like **diffusion-limited growth** and **Ostwald ripening** contribute to nanoparticle growth.
- **Machine Learning in Nanotechnology**: Implementing **Random Forest** and **Gradient Boosting** to predict nanoparticle size and surface features.
