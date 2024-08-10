

# Energy Load Prediction using DAEWOO Steel Co. Ltd Data

## Introduction

Energy consumption is crucial in the industrial sector, influencing both operational efficiency and cost management. To optimize energy usage, it's essential to understand the factors driving energy demand. This project focuses on analyzing electricity consumption data from DAEWOO Steel Co. Ltd, a prominent steel manufacturer in Gwangyang, South Korea. The objective is to predict energy load types—Light Load, Medium Load, or Maximum Load—based on various attributes related to electricity usage.

## Dataset

The dataset includes the following attributes:
- **date**: The date of the observation.
- **Usage_kWh**: Electricity consumption in kilowatt-hours.
- **Lagging_Current_Reactive_Power_kVarh**: Reactive power in kilovolt-ampere reactive hours (lagging).
- **Leading_Current_Reactive_Power_kVarh**: Reactive power in kilovolt-ampere reactive hours (leading).
- **CO2(tCO2)**: Carbon dioxide emissions in tons.
- **Lagging_Current_Power_Factor**: Power factor for lagging current.
- **Leading_Current_Power_Factor**: Power factor for leading current.
- **NSM**: Not specified in the description.
- **WeekStatus**: Indicator for the week status.
- **Day_of_week**: Day of the week.
- **Load_Type**: Target variable representing the load type (Light Load, Medium Load, Maximum Load).

## Project Overview

The goal of this project is to predict the **Load_Type** based on the given data. We employ various machine learning models to classify the load types and gain insights into the factors influencing energy demand.

### Steps Taken

1. **Exploratory Data Analysis (EDA)**: Investigated the dataset to understand its structure and identify key features.
2. **Data Preprocessing**: Cleaned and prepared the data for modeling, including handling missing values and feature scaling.
3. **Modeling**:
   - **Logistic Regression**
   - **Naive Bayes**
   - **Random Forest**
   - **Support Vector Machine (SVM)**
   - **Decision Tree**
   - **Artificial Neural Network (ANN)**
4. **Statistical Analysis**: Conducted deep statistical analysis of model coefficients to understand feature importance.
5. **Hyperparameter Tuning**: Optimized model performance through hyperparameter adjustments.

### Objectives

- **Factor Impact Analysis**: Determine which factors significantly impact the classification of energy load types.
- **Load Trigger Identification**: Identify conditions that lead to a maximum energy load.
- **Efficiency Assessment**: Evaluate energy consumption efficiency and provide insights for better management.

## Results and Discussion

The analysis reveals key insights into the factors driving different energy load types. By understanding these patterns, we offer actionable recommendations for optimizing energy management, contributing to cost savings and sustainability.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/energy-load-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd energy-load-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to execute the analysis and model predictions:
```bash
python main.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Thanks to DAEWOO Steel Co. Ltd for providing the dataset and to all contributors and resources that helped in the completion of this project.

---

Feel free to adjust or expand upon any section as needed!
