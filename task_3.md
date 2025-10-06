# STC Jawwy - Recommendation Engine

## Project Description

This Jupyter Notebook (`stcTV_T3_Sample_Answers.ipynb`) implements a movie and TV show recommendation engine using a dataset provided by STC Jawwy. The primary goal is to recommend programs to users based on their viewing history and ratings, leveraging collaborative filtering techniques.

## Dataset

The project utilizes the `stc- Jawwy TV Data Set_T3.xlsb` dataset. This dataset contains comprehensive details about STC Jawwy customers, including:

*   `user_id_maped`: Anonymized user ID.
*   `program_name`: The name of the movie or TV show.
*   `rating`: User's rating for the program.
*   `date_`: The date of the rating/viewing.
*   `program_genre`: The genre of the program.

The dataset has a shape of (1,048,575, 5) and does not contain any null values, ensuring data integrity for analysis.

## Dependencies

The following Python libraries are required to run this notebook:

*   `pandas`: For data manipulation and analysis.
*   `pyxlsb`: To read the `.xlsb` Excel file format.
*   `numpy`: For numerical operations, especially array manipulation.
*   `scipy`: Specifically `scipy.sparse` for creating sparse matrices.
*   `scikit-learn`: Specifically `sklearn.neighbors.NearestNeighbors` for implementing the K-Nearest Neighbors algorithm.
*   `matplotlib.pyplot`: For basic plotting and visualizations (though not extensively used in the provided sample).
*   `plotly`, `plotly.express`, `plotly.graph_objects`: For interactive visualizations (imported but not explicitly used for output in the sample).

To install these dependencies, you can use pip:

```bash
pip install pandas pyxlsb numpy scikit-learn matplotlib plotly
```

## Usage

To run this project, follow these steps:

1.  **Ensure Dependencies are Installed**: Make sure all the required libraries listed above are installed in your Python environment.
2.  **Place Dataset**: Ensure the `stc- Jawwy TV Data Set_T3.xlsb` file is in the same directory as the Jupyter Notebook, or update the path in the notebook accordingly.
3.  **Open Jupyter Notebook**: Launch Jupyter Notebook and open `stcTV_T3_Sample_Answers.ipynb`.
4.  **Execute Cells**: Run all cells in the notebook sequentially.

### Key Steps in the Notebook:

*   **Data Loading**: The notebook loads the dataset using `pd.read_excel`.
*   **Data Exploration**: Basic data exploration is performed using `dataframe.shape`, `dataframe.head()`, `dataframe.describe()`, and `dataframe.isnull().any()`.
*   **Feature Engineering**: A pivot table (`movie_features_df`) is created to transform the data into a user-item matrix, where rows represent programs and columns represent users, with values being the ratings. Missing values are filled with 0.
*   **Model Training**: A `NearestNeighbors` model from `scikit-learn` is initialized with `cosine` metric and `brute` algorithm, and then fitted to the sparse matrix representation of the `movie_features_df`.
*   **Recommendation Generation**: The notebook demonstrates how to get recommendations for a specific program (e.g., 'Moana') by finding its nearest neighbors in the feature space. The output lists the top 5 recommended programs along with their cosine distances.

## Recommendation Logic

The recommendation engine uses a **collaborative filtering** approach based on the K-Nearest Neighbors (KNN) algorithm. It works by:

1.  **Creating a User-Item Matrix**: This matrix represents programs as rows and users as columns, with cell values indicating the rating a user gave to a program. Programs not rated by a user are filled with zeros.
2.  **Calculating Similarity**: The `NearestNeighbors` model, configured with `cosine` similarity, calculates the similarity between programs based on user ratings. Programs with similar rating patterns across users are considered similar.
3.  **Generating Recommendations**: For a given program, the model identifies the `k` most similar programs. These similar programs are then recommended to users who have shown interest in the initial program.

## Author

Manus AI

