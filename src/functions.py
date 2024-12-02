"""Various functions, classes and variables to assist throughout a data science project.

It includes functionality for:
- Data cleaning: displaying missing values and checking for duplicates.
- Machine learning: Evaluating different models and plotting confusion matrices.

Typical usage examples:
    functions.DataCleaning(df)
    functions.MachineLearning(df)
    functions.phik_between_predictors(df)
    functions.plot_confusion(y_test, y_pred)
"""
import phik
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import ClassifierMixin
from typing import Dict, List, Optional, Union, Type, Tuple
from sklearn.metrics import (
    make_scorer,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)


class DataCleaning:
    """Performs data cleaning tasks including duplicate detection, missing value analysis, and univariate distribution plotting."""

    def __init__(
        self, df: pd.DataFrame, preview: bool = True, display_rows: int = 5
    ) -> None:
        """Initialize DataCleaning with a DataFrame, displaying initial preview and data types if specified."""
        self.df = df.copy()
        if preview:
            display(self.df.head(display_rows))
            rows, columns = self.df.shape
            print(f"DataFrame consists of {rows} rows and {columns} columns.\n")
            print(f"Original data types:\n{self.df.dtypes.value_counts()}\n")
        self._convert_dtypes()

    def _convert_dtypes(self) -> None:
        """Downcast data types of numeric columns to reduce memory usage."""
        for column in self.df:
            if self.df[column].dtype == "float64":
                self.df[column] = pd.to_numeric(self.df[column], downcast="float")
            elif self.df[column].dtype == "int64":
                self.df[column] = pd.to_numeric(self.df[column], downcast="integer")
            elif self.df[column].nunique() == 2 and self.df[column].dtype == "object":
                self.df[column], _ = pd.factorize(self.df[column])
                self.df[column] = self.df[column].astype("int8")
            elif self.df[column].dtype == "object":
                self.df[column] = self.df[column].astype("category")

    def duplicates(self, *column_names: Union[str, list]) -> None:
        """Print the count and percentage of duplicate rows based on specified columns."""
        for column in column_names:
            if isinstance(column, str):
                duplicate_count = self.df.duplicated(subset=column).sum()
                duplicate_percentage = (duplicate_count / len(self.df)) * 100
                print(
                    f'Duplicates by "{column}": {duplicate_count}, {duplicate_percentage:.2f}%'
                )
            elif isinstance(column, list):
                duplicate_count = self.df.duplicated(column).sum()
                duplicate_percentage = (duplicate_count / len(self.df)) * 100
                print(
                    f"Duplicates by {column}: {duplicate_count}, {duplicate_percentage:.2f}%"
                )

    def missing_values(self) -> None:
        """Display missing values per feature and plot missing value percentages."""
        missing_cells = self.df.isnull()
        percentage_of_missing = missing_cells.mean() * 100
        plt.figure(figsize=(22, 6))
        sns.barplot(
            x=percentage_of_missing.index, y=percentage_of_missing.values, zorder=2
        )
        plt.xticks(rotation=90)
        plt.title("Percentage of Missing Values per Feature")
        plt.ylabel("Percentage of Missing Values")
        plt.xlabel("Features")
        plt.show()

        missing_in_categorical = self.df.select_dtypes("category").isna().sum()
        print(
            f"Number of missing values in categorical features: \n{missing_in_categorical}"
        )

        print(
            f"\nPercentage of rows with missing entries: {(missing_cells.any(axis=1).mean()) * 100:.2f}%"
        )
        rows_with_multiple_missing = (missing_cells.sum(axis=1) > 1).mean() * 100
        print(
            f"Percentage of rows with more than one missing value: {rows_with_multiple_missing:.2f}%"
        )
        print(
            f"Percentage of columns with missing entries: {(missing_cells.any().mean()) * 100:.2f}%"
        )
        print(f"Number of columns with missing entries: {(missing_cells.any().sum())}")

    def top_missing_value_groups(self, num_largest_groups: int = 5) -> None:
        """Display groups of features with the highest counts of missing values."""
        missing_values_in_columns = self.df.isnull().sum()
        columns_with_missing_values = missing_values_in_columns[
            missing_values_in_columns > 0
        ]
        matching_counts_of_missing = columns_with_missing_values.value_counts().head(
            num_largest_groups
        )
        for index, value in matching_counts_of_missing.items():
            print(f"Group of {value} features with {index} missing values:")
            print(
                missing_values_in_columns[
                    missing_values_in_columns == index
                ].index.to_list()
            )
            print()

    def numeric_distributions(self) -> None:
        """Plot distributions of all numeric features in the DataFrame."""
        numeric_features = self.df.select_dtypes(include=["number"])
        num_features = len(numeric_features.columns)
        plots_per_row = 4
        num_rows = math.ceil(num_features / plots_per_row)
        fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(15, 3 * num_rows))
        axes = axes.flatten()
        for i, col in enumerate(numeric_features.columns):
            sns.histplot(
                self.df[col], ax=axes[i], zorder=2, edgecolor="white", linewidth=0.5
            )
            axes[i].set_title(f"Distribution of {col}", fontsize=11)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    def categorical_distributions(self) -> None:
        """Plot distributions of all categorical features in the DataFrame."""
        categorical_features = self.df.select_dtypes(include=["category", "object"])
        num_features = len(categorical_features.columns)
        plots_per_row = 3
        num_rows = math.ceil(num_features / plots_per_row)
        fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(15, 4 * num_rows))
        axes = axes.flatten()
        for i, col in enumerate(categorical_features.columns):
            sns.countplot(y=self.df[col], ax=axes[i], zorder=2)
            axes[i].set_title(f"Distribution of {col}", fontsize=11)
            axes[i].set_ylabel(col)
            axes[i].set_xlabel("Frequency")
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()


class MachineLearning:
    """Handles feature selection, engineering, and model evaluation for machine learning tasks."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        id: Optional[List[str]] = None,
        *,
        display: bool = True,
    ) -> None:
        """Initializes the MachineLearning class with a dataset, target variable, and optional ID columns."""
        self.df = df.copy()
        self.target = target
        self.id = id
        self.phik_target = None
        self.phik_predictors = None
        self._convert_dtypes()
        self._phik_target_predictor(display)

    def _convert_dtypes(self):
        """Downcasts data types of all features where possible to improve memory efficiency and boost processing speed."""
        id_columns = self.id if self.id is not None else []
        for column in self.df.drop(columns=id_columns):
            if self.df[column].dtype == "float64":
                self.df[column] = pd.to_numeric(self.df[column], downcast="float")
            elif self.df[column].dtype == "int64":
                self.df[column] = pd.to_numeric(self.df[column], downcast="integer")
            elif self.df[column].nunique() == 2 and self.df[column].dtype == "object":
                self.df[column], _ = pd.factorize(self.df[column])
                self.df[column] = self.df[column].astype("int8")
            elif self.df[column].dtype == "object":
                self.df[column] = self.df[column].astype("category")

    def _phik_target_predictor(self, display: bool):
        """Calculates phik correlation coefficient between the target variable and each predictor."""
        phik_values = []
        numeric_features = self.df.select_dtypes(include=["number"])
        for column in self.df.drop(columns=self.id):
            if column != self.target:
                coeff_calculation_df = self.df[[self.target, column]]
                matrix = coeff_calculation_df.phik_matrix(
                    interval_cols=numeric_features
                )
                try:
                    phik_value = matrix.iloc[0, 1]
                except:
                    phik_value = 0
                phik_values.append((column, phik_value))
        self.phik_target = pd.DataFrame(
            phik_values, columns=["Feature", "Phik Coefficient"]
        ).set_index("Feature")
        self.phik_target = self.phik_target.sort_values(
            by="Phik Coefficient", ascending=False
        )

        if display:
            max_features_per_plot = 160
            num_features = len(self.phik_target)
            plotted_features = min(max_features_per_plot, num_features)
            chunk = self.phik_target.iloc[:plotted_features]
            plt.figure(figsize=(22, 6))
            sns.barplot(x=chunk.index, y=chunk["Phik Coefficient"], zorder=2)
            plt.xticks(rotation=90)
            plt.title(
                f"Phik Correlation Coefficient for Top {plotted_features} Features (Out of {num_features} Total)"
            )
            plt.ylabel("Phik Coefficient")
            plt.xlabel("Features")
            plt.show()

    def evaluate_models(
        self,
        threshold: float,
        models: Dict[str, Type[ClassifierMixin]],
        numeric_imputer: Optional[Union[SimpleImputer, str]],
        categorical_encoding: Optional[Union[TargetEncoder, OneHotEncoder]],
        names_of_highly_correlated: Optional[List[str]] = [],
    ) -> None:
        """Train varous models with ability to do feature selection based on phik correlation coefficient."""
        X = self.df.drop(
            columns=[self.target] + self.id + names_of_highly_correlated
        ).copy()
        y = self.df[self.target].copy()
        y_encoded = LabelEncoder().fit_transform(y)
        features_below_threshold = self.phik_target[
            self.phik_target["Phik Coefficient"] < threshold
        ].index
        X.drop(columns=features_below_threshold, inplace=True)
        scoring = {
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
            "f1": make_scorer(f1_score, pos_label=1),
            "precision": make_scorer(precision_score, pos_label=1),
            "recall": make_scorer(recall_score, pos_label=1),
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
        balance_ratio = y.value_counts().iloc[0] / y.value_counts().iloc[1]
        balanced_weights = [1, balance_ratio]
        models_with_params = {}
        for name, model_class in models.items():
            if name == "LogisticRegression":
                models_with_params[name] = model_class(
                    random_state=5, class_weight="balanced"
                )
            elif name == "RandomForest":
                models_with_params[name] = model_class(
                    random_state=5, class_weight="balanced"
                )
            elif name == "XGBoost":
                models_with_params[name] = model_class(
                    random_state=5,
                    scale_pos_weight=balance_ratio,
                    objective="binary:logistic",
                )
            elif name == "CatBoost":
                models_with_params[name] = model_class(
                    random_state=5,
                    class_weights=balanced_weights,
                    verbose=0,
                    objective="Logloss",
                )
            elif name == "LightGBM":
                models_with_params[name] = model_class(
                    random_state=5,
                    class_weight="balanced",
                    verbose=0,
                    objective="binary",
                )

        numeric_features = X.select_dtypes(include="number").columns.tolist()
        categorical_features = X.select_dtypes(include="category").columns.tolist()
        preprocessor_tree_based = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", numeric_imputer),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", categorical_encoding),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )
        preprocessor_linear = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", RobustScaler()),
                            ("yeo_johnson", PowerTransformer(method="yeo-johnson")),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(drop="first", handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )
        results = []
        for model_name, single_model in models_with_params.items():
            if model_name == "LogisticRegression":
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor_linear),
                        ("classifier", single_model),
                    ]
                )
            else:
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor_tree_based),
                        ("classifier", single_model),
                    ]
                )
            cv_scores = cross_validate(
                pipeline,
                X,
                y_encoded,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
            )
            result = {
                "model": model_name,
                "roc_auc": cv_scores["test_roc_auc"].mean(),
                "train_roc_auc": cv_scores["train_roc_auc"].mean(),
                "accuracy": cv_scores["test_accuracy"].mean(),
                "f1": cv_scores["test_f1"].mean(),
                "precision": cv_scores["test_precision"].mean(),
                "recall": cv_scores["test_recall"].mean(),
            }
            results.append(result)
        results_df = pd.DataFrame(results).set_index("model")
        results_df.index.name = None
        display(results_df)
        print(
            f"Out of {self.df.shape[1]-2} original predictors, {X.shape[1]} were used.\n"
        )

    def phik_between_predictors(
        self, threshold: float, highly_correlated: List[str] = [], display_rows: int = 5
    ) -> None:
        """Calculate phik correlation coefficient between predictors."""
        X = self.df.drop(columns=[self.target] + self.id + highly_correlated).copy()
        features_below_threshold = self.phik_target[
            self.phik_target["Phik Coefficient"] < threshold
        ].index
        X.drop(columns=features_below_threshold, inplace=True)
        numeric_features = X.select_dtypes(include=["number"])
        phik_matrix = X.phik_matrix(interval_cols=numeric_features)
        self.phik_predictors = phik_matrix.unstack().sort_values(ascending=False)
        mask = np.tril(np.ones(phik_matrix.shape), k=-1).astype(bool)
        self.phik_predictors = (
            phik_matrix.where(mask).unstack().sort_values(ascending=False).dropna()
        )
        display(self.phik_predictors.head(display_rows))

    def display_phik_between_predictors(self, display_rows: int = 10):
        """Display calculated phik correlation coefficients."""
        display(self.phik_predictors.head(display_rows))

    def return_selected_features(
        self, threshold: float, highly_correlated: List[str] = []
    ):
        """Return a dataframe with usefull features after feature selection."""
        features_below_threshold = self.phik_target[
            self.phik_target["Phik Coefficient"] < threshold
        ].index.to_list()
        return self.df.drop(columns=highly_correlated + features_below_threshold)


def phik_between_predictors(
    df: pd.DataFrame,
    highly_correlated_to_drop: Optional[List[str]] = None,
    display_rows: int = 5,
) -> None:
    """Calculate and display Phik correlation coefficient between predictors."""
    if highly_correlated_to_drop is None:
        highly_correlated_to_drop = []
    df_copy = df.copy()
    df_copy.drop(columns=highly_correlated_to_drop, inplace=True)
    numeric_features = df_copy.select_dtypes(include=["number"])
    phik_matrix = df_copy.phik_matrix(interval_cols=numeric_features)
    phik_predictors = phik_matrix.unstack().sort_values(ascending=False)
    mask = np.tril(np.ones(phik_matrix.shape), k=-1).astype(bool)
    phik_predictors = (
        phik_matrix.where(mask).unstack().sort_values(ascending=False).dropna()
    )
    display(phik_predictors.head(display_rows))


def mean_mode_aggregation(
    df: pd.DataFrame, id: str, to_exclude: Optional[List[str]] = None
) -> pd.DataFrame:
    """Perform mean aggregation on numeric columns, and mode on categorical."""
    if to_exclude is None:
        to_exclude = []
    numeric_features = (
        df.select_dtypes(include="number")
        .drop(columns=[id] + to_exclude, errors="ignore")
        .columns
    )
    categorical_features = (
        df.select_dtypes(exclude="number")
        .drop(columns=to_exclude, errors="ignore")
        .columns
    )

    agg_dict = {
        **{col: ["mean"] for col in numeric_features},
        **{
            col: lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA
            for col in categorical_features
        },
    }
    df_aggregated = df.groupby(id, as_index=False).agg(agg_dict)
    df_aggregated.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in df_aggregated.columns
    ]
    return df_aggregated


def mean_aggregation(
    df: pd.DataFrame, id: str, to_exclude: Optional[List[str]] = None
) -> pd.DataFrame:
    """Perform mean aggregation on numeric columns."""
    if to_exclude is None:
        to_exclude = []
    numeric_features = (
        df.select_dtypes(include="number")
        .drop(columns=[id] + to_exclude, errors="ignore")
        .columns
    )
    agg_dict = {
        **{col: ["mean"] for col in numeric_features},
    }
    df_aggregated = df.groupby(id, as_index=False).agg(agg_dict)
    df_aggregated.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in df_aggregated.columns
    ]
    return df_aggregated


def min_max_mean_mode_aggregation(
    df: pd.DataFrame, id: str, to_exclude: Optional[List[str]] = None
) -> pd.DataFrame:
    """Perform min, max, and mean aggregation on numeric columns, and mode on categorical."""
    if to_exclude is None:
        to_exclude = []
    numeric_features = (
        df.select_dtypes(include="number")
        .drop(columns=[id] + to_exclude, errors="ignore")
        .columns
    )
    categorical_features = (
        df.select_dtypes(exclude="number")
        .drop(columns=to_exclude, errors="ignore")
        .columns
    )
    agg_dict = {
        **{col: ["mean", "min", "max"] for col in numeric_features},
        **{
            col: lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA
            for col in categorical_features
        },
    }
    df_aggregated = df.groupby(id, as_index=False).agg(agg_dict)
    df_aggregated.columns = [
        "_".join([col[0], col[1].upper()]).strip("_")
        if isinstance(col, tuple)
        else f"{col}_MODE"
        if col in categorical_features
        else col
        for col in df_aggregated.columns
    ]
    df_aggregated.columns = [
        col.replace("<LAMBDA>", "MODE")
        if isinstance(col, str)
        else "_".join([col[0], col[1].upper()]).strip("_")
        for col in df_aggregated.columns
    ]
    return df_aggregated


def mean_aggregation_of_recent(
    df: pd.DataFrame, id: str, column_with_time: str, n_most_recent_values: int
) -> pd.DataFrame:
    """Compute the mean of the n most recent entries the given ID."""
    df_copy = df.copy()
    sorted_df = df_copy.sort_values([id, column_with_time], ascending=[True, False])
    recent_df = sorted_df.groupby(id).head(n_most_recent_values)
    averaged_df = recent_df.groupby(id).mean(numeric_only=True)
    if column_with_time in averaged_df.columns:
        averaged_df = averaged_df.drop(columns=column_with_time)
    averaged_df.columns = [f"RECENT_{col}_MEAN" for col in averaged_df.columns]
    return averaged_df


def downcasting(df: pd.DataFrame, id: Optional[str] = None) -> pd.DataFrame:
    """Downcast data types of all features where possible to improve memory efficiency and boost processing speed."""
    df_copy = df.copy()
    columns_to_downcast = df_copy.columns.drop(id) if id else df_copy.columns
    for column in columns_to_downcast:
        if df_copy[column].dtype == "float64":
            df_copy[column] = pd.to_numeric(df_copy[column], downcast="float")
        elif df_copy[column].dtype == "int64":
            df_copy[column] = pd.to_numeric(df_copy[column], downcast="integer")
        elif df_copy[column].nunique() == 2 and df_copy[column].dtype == "object":
            df_copy[column], _ = pd.factorize(df_copy[column])
            df_copy[column] = df_copy[column].astype("int8")
        elif df_copy[column].dtype == "object":
            df_copy[column] = df_copy[column].astype("category")
    return df_copy


def plot_confusion_matrix(y_test: pd.Series, y_test_pred: np.ndarray) -> None:
    """Plot the confusion matrix for a test dataset."""
    colors = ["#FFFFFF", "#DD8452"]
    n_bins = [0, 1]
    cmap = LinearSegmentedColormap.from_list("custom_blue", list(zip(n_bins, colors)))
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(3, 3))
    sns.heatmap(
        conf_matrix_test,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        linewidths=0.7,
    )
    plt.grid(False)
    plt.yticks(rotation=0)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Confusion Matrix for the Test Set")
    plt.tight_layout()
    plt.show()


def matching_ids(
    df1: pd.DataFrame, df2: pd.DataFrame, df1_name: str, df2_name: str, id_column: str
) -> None:
    """Checks the match rate of IDs between two DataFrames and prints it out."""
    match_rate1 = df1[id_column].isin(df2[id_column]).mean() * 100
    match_rate2 = df2[id_column].isin(df1[id_column]).mean() * 100
    print(f"{match_rate1:.2f}% of IDs in {df1_name} are also present in {df2_name}.")
    print(f"{match_rate2:.2f}% of IDs in {df2_name} are also present in {df1_name}.")


def plot_final_evaluation_curves(y_test: np.ndarray, y_pred_prob: np.ndarray) -> None:
    """Plot ROC and Precision-Recall curves, and threshold performance curves for evaluation."""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(
        fpr,
        tpr,
        lw=2,
        color="#DD8452",
        label=f"Positive class ROC curve (area = {roc_auc_score(y_test, y_pred_prob):.2f})",
        zorder=2,
    )
    y_pred_prob_neg = 1 - y_pred_prob
    fpr_neg, tpr_neg, _ = roc_curve(1 - y_test, y_pred_prob_neg)
    plt.plot(
        fpr_neg,
        tpr_neg,
        lw=2,
        color="#4C72B0",
        label=f"Negative class ROC curve (area = {roc_auc_score(1 - y_test, y_pred_prob_neg):.2f})",
        zorder=2,
    )
    plt.plot([0, 1], [0, 1], color="Grey", lw=2, linestyle="--", zorder=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.subplot(1, 2, 2)
    plt.plot(
        recall,
        precision,
        lw=2,
        color="#DD8452",
        label=f"Positive class PR curve (area = {average_precision_score(y_test, y_pred_prob):.2f})",
        zorder=2,
    )
    precision_neg, recall_neg, _ = precision_recall_curve(1 - y_test, y_pred_prob_neg)
    plt.plot(
        recall_neg,
        precision_neg,
        lw=2,
        color="#4C72B0",
        label=f"Negative class PR curve (area = {average_precision_score(1 - y_test, y_pred_prob_neg):.2f})",
        zorder=2,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    thresholds = np.arange(0.0, 1.0, 0.01)
    precisions = []
    recalls = []
    f1s = []
    for threshold in thresholds:
        y_pred_threshold = (y_pred_prob >= threshold).astype(int)
        precisions.append(precision_score(y_test, y_pred_threshold, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_threshold))
        f1s.append(f1_score(y_test, y_pred_threshold))
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, recalls, label="recall", color="#1f77b4")
    plt.plot(thresholds, precisions, label="precision", color="#2ca02c")
    plt.plot(thresholds, f1s, label="f1", color="#ff7f0e")
    plt.xlabel("Cutoff")
    plt.ylabel("Score")
    plt.title("Threshold Performance Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def permutation_median_diff(
    df: pd.DataFrame, target_feature: str, predictor: str, num_samples: int = 500
) -> float:
    """Perform a permutation hypothesis test to compare the difference in medians between two groups."""
    categories = df[target_feature].unique()
    data_cat1 = df[df[target_feature] == categories[0]][predictor]
    data_cat2 = df[df[target_feature] == categories[1]][predictor]
    observed_diff = np.median(data_cat1) - np.median(data_cat2)
    combined = np.concatenate([data_cat1, data_cat2])
    boot_diffs = []
    for _ in range(num_samples):
        boot_cat1 = np.random.choice(combined, size=len(data_cat1), replace=True)
        boot_cat2 = np.random.choice(combined, size=len(data_cat2), replace=True)
        boot_diff = np.median(boot_cat1) - np.median(boot_cat2)
        boot_diffs.append(boot_diff)
    p_value = np.sum(np.abs(boot_diffs) >= np.abs(observed_diff)) / num_samples
    return p_value


def numeric_relationships(df: pd.DataFrame, target_feature: str) -> None:
    """Analyze the relationship between numeric predictors and a target feature."""
    selected_columns = df.select_dtypes(include=["number"]).columns.tolist()
    selected_columns.append(target_feature)
    temp_df = df[selected_columns].copy()
    for predictor in selected_columns:
        if predictor != target_feature:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"Distribution of {predictor} by {target_feature}",
                fontsize=13,
            )
            sns.histplot(
                data=temp_df,
                x=predictor,
                hue=target_feature,
                multiple="stack",
                edgecolor="white",
                ax=axs[0],
                zorder=2,
                linewidth=0.5,
                alpha=0.98,
            )
            axs[0].set_xlabel(predictor)
            axs[0].set_ylabel("Frequency")
            sns.boxplot(
                data=temp_df,
                y=predictor,
                x=target_feature,
                ax=axs[1],
                zorder=2,
            )
            axs[1].set_ylabel(predictor)
            axs[1].set_xlabel(target_feature)
            plt.tight_layout()
            plt.show()
            medians = temp_df.groupby(target_feature)[predictor].median()
            print(f"Medians for {predictor} by {target_feature}:")
            for group, median in medians.items():
                print(f"{group}: {median:.2f}")
            p_value = permutation_median_diff(df, target_feature, predictor)
            print(f"Bootstrap hypothesis test p-value: {p_value:.2f}")


def categorical_relationships(df: pd.DataFrame, target_feature: str) -> None:
    """Analyze the relationship between categorical predictors and a target feature."""
    selected_columns = df.select_dtypes(include=["category", "object"]).columns.tolist()
    if target_feature not in selected_columns:
        selected_columns.append(target_feature)
    temp_df = df[selected_columns].copy()
    for predictor_name in temp_df.columns:
        if predictor_name != target_feature:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"Distribution of {predictor_name} by {target_feature}",
                fontsize=13,
            )
            crosstab = pd.crosstab(temp_df[predictor_name], temp_df[target_feature])
            crosstab.plot(
                kind="barh",
                stacked=True,
                ax=axs[0],
                edgecolor="white",
                linewidth=0.5,
            )
            axs[0].grid(axis="x")
            axs[0].set_xlabel(predictor_name)
            axs[0].set_ylabel("Count")
            for patch in axs[0].patches:
                patch.set_zorder(2)
            crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
            crosstab_normalized.plot(
                kind="barh", stacked=True, ax=axs[1], edgecolor="white", linewidth=0.5
            )
            axs[1].grid(axis="x")
            axs[1].set_ylabel("Proportion")
            axs[1].set_xlabel(predictor_name)
            axs[1].legend().set_visible(False)
            for patch in axs[1].patches:
                patch.set_zorder(2)
            plt.tight_layout()
            plt.show()
            _, p_value, _, _ = chi2_contingency(crosstab)
            print(
                f"{predictor_name} - {target_feature}:\nChi-Square test p-value: {p_value:.2f}\n"
            )


def feature_engineering_pipeline(
    application_new: pd.DataFrame,
    previous_application_new: pd.DataFrame,
    bureau_new: pd.DataFrame,
    bureau_balance_new: pd.DataFrame,
    installments_payments_new: pd.DataFrame,
    pos_cash_balance_new: pd.DataFrame,
    credit_card_balance_new: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Perform feature engineering on the provided dataframes."""
    # Create copies of the input dataframes
    application = application_new.copy()
    test_id_curr = application["SK_ID_CURR"]
    previous_application = previous_application_new[
        previous_application_new["SK_ID_CURR"].isin(test_id_curr)
    ].copy()
    bureau = bureau_new[bureau_new["SK_ID_CURR"].isin(test_id_curr)].copy()
    bureau_balance = bureau_balance_new[
        bureau_balance_new["SK_ID_BUREAU"].isin(bureau["SK_ID_BUREAU"])
    ].copy()
    installments_payments = installments_payments_new[
        installments_payments_new["SK_ID_CURR"].isin(test_id_curr)
    ].copy()
    pos_cash_balance = pos_cash_balance_new[
        pos_cash_balance_new["SK_ID_CURR"].isin(test_id_curr)
    ].copy()
    credit_card_balance = credit_card_balance_new[
        credit_card_balance_new["SK_ID_CURR"].isin(test_id_curr)
    ].copy()

    # Application
    application["DEBT_TO_INCOME_RATIO"] = application["AMT_CREDIT"] / application[
        "AMT_INCOME_TOTAL"
    ].replace(0, np.nan)
    application["INCOME_STABILITY"] = application["DAYS_EMPLOYED"].replace(
        {365243: np.nan}
    ) / application["DAYS_BIRTH"].replace(0, np.nan)
    application["CREDIT_TERM_TO_AGE_RATIO"] = application["AMT_ANNUITY"] / application[
        "AMT_CREDIT"
    ].replace(0, np.nan)
    application["INCOME_PER_FAMILY_MEMBER"] = application[
        "AMT_INCOME_TOTAL"
    ] / application["CNT_FAM_MEMBERS"].replace(0, np.nan)
    application["POPULATION_DENSITY_RISK"] = application[
        "REGION_POPULATION_RELATIVE"
    ] * application["REGION_RATING_CLIENT"].replace(0, np.nan)
    application["CREDIT_TO_ANNUITY_RATIO"] = application["AMT_CREDIT"] / application[
        "AMT_ANNUITY"
    ].replace(0, np.nan)
    application["EMPLOYMENT_TO_REGISTRATION_RATIO"] = application[
        "DAYS_EMPLOYED"
    ].replace({365243: np.nan}) / application["DAYS_REGISTRATION"].replace(0, np.nan)
    application["ANNUITY_TO_INCOME_RATIO"] = application["AMT_ANNUITY"] / application[
        "AMT_INCOME_TOTAL"
    ].replace(0, np.nan)
    application["CHILDREN_PER_FAMILY_MEMBER"] = application[
        "CNT_CHILDREN"
    ] / application["CNT_FAM_MEMBERS"].replace(0, np.nan)
    application["DAYS_BIRTH_TO_ANNUITY_RATIO"] = application[
        "DAYS_BIRTH"
    ] / application["AMT_ANNUITY"].replace(0, np.nan)
    application["DAYS_BIRTH_TO_CREDIT_RATIO"] = application["DAYS_BIRTH"] / application[
        "AMT_CREDIT"
    ].replace(0, np.nan)
    application["DAYS_BIRTH_TO_INCOME_RATIO"] = application["DAYS_BIRTH"] / application[
        "AMT_INCOME_TOTAL"
    ].replace(0, np.nan)
    application["DAYS_EMPLOYED_TO_ANNUITY_RATIO"] = application[
        "DAYS_EMPLOYED"
    ].replace({365243: np.nan}) / application["AMT_ANNUITY"].replace(0, np.nan)
    application["DAYS_EMPLOYED_TO_CREDIT_RATIO"] = application["DAYS_EMPLOYED"].replace(
        {365243: np.nan}
    ) / application["AMT_CREDIT"].replace(0, np.nan)
    application["DAYS_EMPLOYED_TO_INCOME_RATIO"] = application["DAYS_EMPLOYED"].replace(
        {365243: np.nan}
    ) / application["AMT_INCOME_TOTAL"].replace(0, np.nan)
    application["DAYS_EMPLOYED_TO_GOODS_PRICE_RATIO"] = application[
        "DAYS_EMPLOYED"
    ].replace({365243: np.nan}) / application["AMT_GOODS_PRICE"].replace(0, np.nan)
    application["ANNUITY_TO_GOODS_RATIO"] = application["AMT_ANNUITY"] / application[
        "AMT_GOODS_PRICE"
    ].replace(0, np.nan)
    application["CAR_AGE_TO_AGE_RATIO"] = application["OWN_CAR_AGE"] / application[
        "DAYS_BIRTH"
    ].replace(0, np.nan)
    application["CREDIT_TO_GOODS_RATIO"] = application["AMT_CREDIT"] / application[
        "AMT_GOODS_PRICE"
    ].replace(0, np.nan)
    application["DAYS_BIRTH_TO_OWN_CAR_AGE"] = application["DAYS_BIRTH"] / application[
        "OWN_CAR_AGE"
    ].replace(0, np.nan)
    application["GOODS_PRICE_TO_DAYS_BIRTH"] = application[
        "AMT_GOODS_PRICE"
    ] / application["DAYS_BIRTH"].replace(0, np.nan)
    application["DAYS_REGISTRATION_TO_DAYS_ID_PUBLISH"] = application[
        "DAYS_REGISTRATION"
    ] / application["DAYS_ID_PUBLISH"].replace(0, np.nan)
    correlated_to_drop = [
        "FLAG_MOBIL",
        "NAME_INCOME_TYPE",
        "FLAG_EMP_PHONE",
        "ORGANIZATION_TYPE",
        "INCOME_STABILITY",
        "ENTRANCES_MEDI",
        "BASEMENTAREA_MEDI",
        "ELEVATORS_MEDI",
        "ENTRANCES_MODE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "NONLIVINGAPARTMENTS_MEDI",
        "LANDAREA_MEDI",
        "FLOORSMIN_MEDI",
        "NONLIVINGAPARTMENTS_MEDI",
        "COMMONAREA_MEDI",
        "FLOORSMAX_MEDI",
        "YEARS_BUILD_MEDI",
        "REGION_RATING_CLIENT_W_CITY",
        "APARTMENTS_MEDI",
        "NONLIVINGAPARTMENTS_MODE",
        "LIVINGAPARTMENTS_MEDI",
        "COMMONAREA_MODE",
        "FLOORSMIN_MODE",
        "LIVINGAREA_MEDI",
        "NONLIVINGAREA_MEDI",
        "YEARS_BUILD_MEDI",
        "YEARS_BEGINEXPLUATATION_MEDI",
        "FLOORSMAX_MODE",
        "LANDAREA_MODE",
        "BASEMENTAREA_MODE",
        "YEARS_BUILD_MODE",
        "APARTMENTS_MODE",
    ]
    application.drop(columns=correlated_to_drop, inplace=True, errors="ignore")

    # Previous application
    previous_application = previous_application.loc[
        :, previous_application.isnull().mean() <= 0.95
    ]
    previous_application["CREDIT_TO_APPLICATION_RATIO"] = previous_application[
        "AMT_CREDIT"
    ] / previous_application["AMT_APPLICATION"].replace(0, np.nan)
    previous_application["DOWN_PAYMENT_TO_CREDIT_RATIO"] = previous_application[
        "AMT_DOWN_PAYMENT"
    ] / previous_application["AMT_CREDIT"].replace(0, np.nan)
    previous_application["ANNUITY_TO_CREDIT_RATIO"] = previous_application[
        "AMT_ANNUITY"
    ] / previous_application["AMT_CREDIT"].replace(0, np.nan)
    previous_application["GOODS_PRICE_TO_CREDIT_RATIO"] = previous_application[
        "AMT_GOODS_PRICE"
    ] / previous_application["AMT_CREDIT"].replace(0, np.nan)
    previous_application["PAYMENT_TO_GOODS_PRICE_RATIO"] = previous_application[
        "AMT_ANNUITY"
    ] / previous_application["AMT_GOODS_PRICE"].replace(0, np.nan)
    previous_application["DOWN_PAYMENT_TO_GOODS_PRICE_RATIO"] = previous_application[
        "AMT_DOWN_PAYMENT"
    ] / previous_application["AMT_GOODS_PRICE"].replace(0, np.nan)
    previous_application["DECISION_TO_FIRST_DUE_RATIO"] = previous_application[
        "DAYS_DECISION"
    ] / previous_application["DAYS_FIRST_DUE"].replace(0, np.nan)
    previous_application["FIRST_DRAWING_TO_TERMINATION_RATIO"] = previous_application[
        "DAYS_FIRST_DRAWING"
    ] / previous_application["DAYS_TERMINATION"].replace(0, np.nan)
    previous_application["FIRST_DUE_TO_LAST_DUE_RATIO"] = previous_application[
        "DAYS_FIRST_DUE"
    ] / previous_application["DAYS_LAST_DUE"].replace(0, np.nan)
    previous_application["INSTALLMENTS_TO_CREDIT_RATIO"] = previous_application[
        "CNT_PAYMENT"
    ] / previous_application["AMT_CREDIT"].replace(0, np.nan)
    previous_application["APPLICATION_TO_GOODS_PRICE_RATIO"] = previous_application[
        "AMT_APPLICATION"
    ] / previous_application["AMT_GOODS_PRICE"].replace(0, np.nan)
    previous_application["DOWN_PAYMENT_TO_APPLICATION_RATIO"] = previous_application[
        "AMT_DOWN_PAYMENT"
    ] / previous_application["AMT_APPLICATION"].replace(0, np.nan)
    previous_application["DECISION_TO_TERMINATION_RATIO"] = previous_application[
        "DAYS_DECISION"
    ] / previous_application["DAYS_TERMINATION"].replace(0, np.nan)
    previous_application["GOODS_PRICE_TO_DOWN_PAYMENT_RATIO"] = previous_application[
        "AMT_GOODS_PRICE"
    ] / previous_application["AMT_DOWN_PAYMENT"].replace(0, np.nan)
    previous_application["FIRST_DRAWING_TO_FIRST_DUE_RATIO"] = previous_application[
        "DAYS_FIRST_DRAWING"
    ] / previous_application["DAYS_FIRST_DUE"].replace(0, np.nan)
    previous_application["TERMINATION_TO_LAST_DUE_RATIO"] = previous_application[
        "DAYS_TERMINATION"
    ] / previous_application["DAYS_LAST_DUE"].replace(0, np.nan)
    previous_application["APPLICATION_TO_ANNUITY_RATIO"] = previous_application[
        "AMT_APPLICATION"
    ] / previous_application["AMT_ANNUITY"].replace(0, np.nan)
    previous_application["GOODS_PRICE_TO_CREDIT_TERM_RATIO"] = previous_application[
        "AMT_GOODS_PRICE"
    ] / previous_application["CNT_PAYMENT"].replace(0, np.nan)
    previous_application["NAME_CONTRACT_TYPE"] = previous_application[
        "NAME_CONTRACT_TYPE"
    ].replace(
        {
            "Consumer loans": "Consumer_loans",
            "Cash loans": "Cash_loans",
            "Revolving loans": "Revolving_loans",
        }
    )
    numerical_columns = previous_application.select_dtypes(include=["number"]).columns
    agg_dict = {col: ["mean"] for col in numerical_columns}
    agg_dict["SK_ID_PREV"] = "count"
    previous_application_split_aggregated = (
        previous_application.groupby(
            ["SK_ID_CURR", "NAME_CONTRACT_TYPE"], observed=True
        )
        .agg(agg_dict)
        .reset_index()
    )
    previous_application_split_aggregated.columns = [
        "_".join(col).strip("_").upper() if isinstance(col, tuple) else col.upper()
        for col in previous_application_split_aggregated.columns
    ]
    previous_application_aggregated = previous_application_split_aggregated.pivot(
        index="SK_ID_CURR", columns="NAME_CONTRACT_TYPE"
    ).reset_index()
    previous_application_aggregated.columns = [
        "_".join(col).strip("_").upper() if isinstance(col, tuple) else col.upper()
        for col in previous_application_aggregated.columns
    ]
    previous_application_aggregated.drop(
        columns=[
            "SK_ID_CURR_MEAN_CASH_LOANS",
            "SK_ID_CURR_MEAN_CONSUMER_LOANS",
            "SK_ID_CURR_MEAN_REVOLVING_LOANS",
        ],
        inplace=True,
        errors="ignore",
    )
    previous_application_all_aggregated = min_max_mean_mode_aggregation(
        previous_application, "SK_ID_CURR", ["SK_ID_PREV"]
    )
    previous_application_aggregated = previous_application_aggregated.merge(
        previous_application_all_aggregated, on="SK_ID_CURR", how="left"
    )
    previous_application_aggregated = previous_application_aggregated.loc[
        :, previous_application_aggregated.isnull().mean() <= 0.95
    ]

    # Bureau balance and bureau
    status_counts = bureau_balance.pivot_table(
        index="SK_ID_BUREAU", columns="STATUS", aggfunc="size", fill_value=0
    )
    status_proportions = status_counts.div(status_counts.sum(axis=1), axis=0)
    bureau_balance["STATUS_num"] = pd.to_numeric(
        bureau_balance["STATUS"], errors="coerce"
    )
    max_status = bureau_balance.groupby("SK_ID_BUREAU")["STATUS_num"].max()
    status_proportions = status_proportions.reset_index()
    status_proportions.columns.name = None
    status_proportions = status_proportions.set_index("SK_ID_BUREAU")

    def count_consecutive_zeros_before_c(status_series: pd.Series) -> int:
        """Count consecutive '0' values in a series until the first occurrence of 'C'."""
        count = 0
        for status in status_series:
            if status == "C":
                break
            elif status == "0":
                count += 1
            else:
                count = 0
        return count

    consecutive_zeros = bureau_balance.groupby("SK_ID_BUREAU")["STATUS"].apply(
        count_consecutive_zeros_before_c
    )
    last_status = bureau_balance[bureau_balance["MONTHS_BALANCE"] == 0].set_index(
        "SK_ID_BUREAU"
    )["STATUS"]
    sorted_bureau_balance = bureau_balance.sort_values(
        ["SK_ID_BUREAU", "MONTHS_BALANCE"], ascending=[True, False]
    )
    recent_bureau_balance = sorted_bureau_balance.groupby("SK_ID_BUREAU").head(5)
    mode_bureau_balance = recent_bureau_balance.groupby("SK_ID_BUREAU")["STATUS"].apply(
        lambda x: x.mode()[0]
    )
    bureau_balance_aggregated = (
        pd.DataFrame(
            {
                "MAX_STATUS": max_status,
                "CONSECUTIVE_ZEROS_BEFORE_C": consecutive_zeros,
                "LAST_STATUS_AT_MOST_RECENT_MONTH": last_status,
                "RECENT_STATUS_MODE": mode_bureau_balance,
            }
        )
        .join(status_proportions, how="left")
        .reset_index()
    )
    bureau_balance_aggregated.rename(
        columns={
            "0": "ZERO",
            "1": "ONE",
            "2": "TWO",
            "3": "THREE",
            "4": "FOUR",
            "5": "FIVE",
        },
        inplace=True,
    )
    bureau_balance_aggregated.rename(
        columns={
            "0": "ZERO",
            "1": "ONE",
            "2": "TWO",
            "3": "THREE",
            "4": "FOUR",
            "5": "FIVE",
        },
        inplace=True,
    )
    bureau["CREDIT_DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau[
        "AMT_CREDIT_SUM"
    ].replace(0, np.nan)
    bureau["CREDIT_OVERDUE_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / bureau[
        "AMT_CREDIT_SUM"
    ].replace(0, np.nan)
    bureau["CREDIT_LIMIT_RATIO"] = bureau["AMT_CREDIT_SUM_LIMIT"] / bureau[
        "AMT_CREDIT_SUM"
    ].replace(0, np.nan)
    bureau["MAX_OVERDUE_TO_CREDIT_RATIO"] = bureau["AMT_CREDIT_MAX_OVERDUE"] / bureau[
        "AMT_CREDIT_SUM"
    ].replace(0, np.nan)
    bureau["PROLONG_TO_CREDIT_RATIO"] = bureau["CNT_CREDIT_PROLONG"] / bureau[
        "AMT_CREDIT_SUM"
    ].replace(0, np.nan)
    bureau["OVERDUE_DAYS_RATIO"] = bureau["CREDIT_DAY_OVERDUE"] / bureau[
        "DAYS_CREDIT"
    ].replace(0, np.nan)
    bureau["CREDIT_DURATION_RATIO"] = bureau["DAYS_CREDIT_ENDDATE"] / bureau[
        "DAYS_CREDIT"
    ].replace(0, np.nan)
    bureau["DAYS_END_TO_DURATION_RATIO"] = bureau["DAYS_ENDDATE_FACT"] / bureau[
        "DAYS_CREDIT"
    ].replace(0, np.nan)
    bureau["CREDIT_ANN_DEBT_RATIO"] = bureau["AMT_ANNUITY"] / bureau[
        "AMT_CREDIT_SUM_DEBT"
    ].replace(0, np.nan)
    bureau["CREDIT_UPDATE_DURATION_RATIO"] = bureau["DAYS_CREDIT_UPDATE"] / bureau[
        "DAYS_CREDIT"
    ].replace(0, np.nan)
    bureau["ANNUITY_TO_CREDIT_RATIO"] = bureau["AMT_ANNUITY"] / bureau[
        "AMT_CREDIT_SUM"
    ].replace(0, np.nan)
    bureau_merged = bureau.merge(
        bureau_balance_aggregated, on="SK_ID_BUREAU", how="left"
    )
    bureau_merged = bureau_merged.drop(columns="SK_ID_BUREAU")
    bureau_merged_aggregated = min_max_mean_mode_aggregation(
        bureau_merged, "SK_ID_CURR", ["CREDIT_CURRENCY"]
    )

    # Installments Payments
    installments_payments_reworked = pd.DataFrame(
        {
            "SK_ID_CURR": installments_payments["SK_ID_CURR"],
            "NUM_INSTALMENT_NUMBER": installments_payments["NUM_INSTALMENT_NUMBER"],
            "DAYS_LATE": installments_payments["DAYS_ENTRY_PAYMENT"]
            - installments_payments["DAYS_INSTALMENT"],
            "NORMALIZED_DAYS_LATE": (
                installments_payments["DAYS_ENTRY_PAYMENT"]
                - installments_payments["DAYS_INSTALMENT"]
            )
            / installments_payments["DAYS_INSTALMENT"].replace(0, np.nan),
            "AMOUNT_DIFFERENCE": installments_payments["AMT_PAYMENT"]
            - installments_payments["AMT_INSTALMENT"],
            "NORMALIZED_AMOUNT_DIFFERENCE": (
                installments_payments["AMT_PAYMENT"]
                - installments_payments["AMT_INSTALMENT"]
            )
            / installments_payments["AMT_INSTALMENT"].replace(0, np.nan),
            "LATE_PAYMENT": (
                installments_payments["DAYS_ENTRY_PAYMENT"]
                > installments_payments["DAYS_INSTALMENT"]
            ).astype(int),
        }
    )
    installments_payments_aggregated = installments_payments_reworked.groupby(
        "SK_ID_CURR"
    ).agg(
        TOTAL_NUM_INSTALLMENTS=("DAYS_LATE", "count"),
        TOTAL_LATE_PAYMENTS=("LATE_PAYMENT", "sum"),
        LATE_PAYMENT_RATIO=("LATE_PAYMENT", lambda x: x.sum() / x.count()),
        MEAN_DAYS_LATE=("DAYS_LATE", "mean"),
        MEAN_NORMALIZED_DAYS_LATE=("NORMALIZED_DAYS_LATE", "mean"),
        MEAN_AMOUNT_DIFFERENCE=("AMOUNT_DIFFERENCE", "mean"),
        MEAN_NORMALIZED_AMOUNT_DIFFERENCE=("NORMALIZED_AMOUNT_DIFFERENCE", "mean"),
    )
    installments_payments_reworked_sorted = installments_payments_reworked.sort_values(
        by=["SK_ID_CURR", "NUM_INSTALMENT_NUMBER"]
    )
    first_five_installments = installments_payments_reworked_sorted.groupby(
        "SK_ID_CURR"
    ).head(5)
    late_in_first_five = (
        first_five_installments.groupby("SK_ID_CURR")["LATE_PAYMENT"]
        .sum()
        .reset_index()
    )
    late_in_first_five.columns = ["SK_ID_CURR", "LATE_PAYMENTS_FIRST_5"]
    installments_payments_aggregated = installments_payments_aggregated.merge(
        late_in_first_five, on="SK_ID_CURR", how="left"
    )
    version_counts = (
        installments_payments.groupby("SK_ID_CURR")["NUM_INSTALMENT_VERSION"]
        .nunique()
        .reset_index()
    )
    version_counts.columns = ["SK_ID_CURR", "NUM_INSTALMENT_VERSIONS"]
    installments_payments_aggregated = installments_payments_aggregated.merge(
        version_counts, on="SK_ID_CURR", how="left"
    )

    # POS cash balance
    pos_cash_balance["INSTALMENTS_COMPLETION_RATIO"] = (
        pos_cash_balance["CNT_INSTALMENT"] - pos_cash_balance["CNT_INSTALMENT_FUTURE"]
    ) / pos_cash_balance["CNT_INSTALMENT"].replace(0, np.nan)
    pos_cash_balance["DPD_TO_REMAINING_INSTALMENTS_RATIO"] = pos_cash_balance[
        "SK_DPD"
    ] / pos_cash_balance["CNT_INSTALMENT_FUTURE"].replace(0, np.nan)
    pos_cash_balance["DPD_DEF_TO_DPD_RATIO"] = pos_cash_balance[
        "SK_DPD_DEF"
    ] / pos_cash_balance["SK_DPD"].replace(0, np.nan)
    pos_cash_balance["AVG_DPD_PER_INSTALMENT"] = pos_cash_balance[
        "SK_DPD"
    ] / pos_cash_balance["CNT_INSTALMENT_FUTURE"].replace(0, np.nan)
    pos_cash_balance["AVG_DPD_DEF_PER_INSTALMENT"] = pos_cash_balance[
        "SK_DPD_DEF"
    ] / pos_cash_balance["CNT_INSTALMENT_FUTURE"].replace(0, np.nan)
    pos_cash_balance["MONTHLY_DELAY_RATIO"] = pos_cash_balance[
        "SK_DPD"
    ] / pos_cash_balance["MONTHS_BALANCE"].replace(0, np.nan)
    pos_cash_balance["ON_TIME_INSTALMENTS_RATIO"] = (
        pos_cash_balance["CNT_INSTALMENT"] - pos_cash_balance["SK_DPD_DEF"]
    ) / pos_cash_balance["CNT_INSTALMENT"].replace(0, np.nan)
    pos_cash_balance = pos_cash_balance.drop(columns="SK_ID_PREV", errors="ignore")
    all_time_pos_cash_balance_aggregated = min_max_mean_mode_aggregation(
        pos_cash_balance, "SK_ID_CURR"
    )
    recent_pos_cash_balance_aggregated = mean_aggregation_of_recent(
        pos_cash_balance, "SK_ID_CURR", "MONTHS_BALANCE", 5
    )
    pos_cash_balance_aggregated = all_time_pos_cash_balance_aggregated.merge(
        recent_pos_cash_balance_aggregated, on="SK_ID_CURR", how="left"
    )
    pos_cash_balance_aggregated = pos_cash_balance_aggregated.drop(
        columns=["MONTHS_BALANCE_MIN", "MONTHS_BALANCE_MEAN"]
    )

    # Credit card balance
    credit_card_balance = credit_card_balance.drop(columns="SK_ID_PREV")
    credit_card_balance["BALANCE_TO_CREDIT_LIMIT_RATIO"] = credit_card_balance[
        "AMT_BALANCE"
    ] / credit_card_balance["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    credit_card_balance["ATM_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO"] = credit_card_balance[
        "AMT_DRAWINGS_ATM_CURRENT"
    ] / credit_card_balance["AMT_DRAWINGS_CURRENT"].replace(0, np.nan)
    credit_card_balance["OTHER_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO"] = credit_card_balance[
        "AMT_DRAWINGS_OTHER_CURRENT"
    ] / credit_card_balance["AMT_DRAWINGS_CURRENT"].replace(0, np.nan)
    credit_card_balance["POS_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO"] = credit_card_balance[
        "AMT_DRAWINGS_POS_CURRENT"
    ] / credit_card_balance["AMT_DRAWINGS_CURRENT"].replace(0, np.nan)
    credit_card_balance["MIN_INSTALLMENT_TO_TOTAL_PAYMENT_RATIO"] = credit_card_balance[
        "AMT_INST_MIN_REGULARITY"
    ] / credit_card_balance["AMT_PAYMENT_TOTAL_CURRENT"].replace(0, np.nan)
    credit_card_balance["PAYMENT_TO_RECEIVABLE_RATIO"] = credit_card_balance[
        "AMT_PAYMENT_CURRENT"
    ] / credit_card_balance["AMT_RECIVABLE"].replace(0, np.nan)
    credit_card_balance[
        "PAYMENT_TOTAL_TO_TOTAL_RECEIVABLE_RATIO"
    ] = credit_card_balance["AMT_PAYMENT_TOTAL_CURRENT"] / credit_card_balance[
        "AMT_TOTAL_RECEIVABLE"
    ].replace(
        0, np.nan
    )
    credit_card_balance["RECEIVABLE_TO_CREDIT_LIMIT_RATIO"] = credit_card_balance[
        "AMT_RECEIVABLE_PRINCIPAL"
    ] / credit_card_balance["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    credit_card_balance["TOTAL_RECEIVABLE_TO_CREDIT_LIMIT_RATIO"] = credit_card_balance[
        "AMT_TOTAL_RECEIVABLE"
    ] / credit_card_balance["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    credit_card_balance["ATM_DRAWINGS_COUNT_RATIO"] = credit_card_balance[
        "CNT_DRAWINGS_ATM_CURRENT"
    ] / credit_card_balance["CNT_DRAWINGS_CURRENT"].replace(0, np.nan)
    credit_card_balance["POS_DRAWINGS_COUNT_RATIO"] = credit_card_balance[
        "CNT_DRAWINGS_POS_CURRENT"
    ] / credit_card_balance["CNT_DRAWINGS_CURRENT"].replace(0, np.nan)
    credit_card_balance["DAYS_PAST_DUE_RATIO"] = credit_card_balance[
        "SK_DPD"
    ] / credit_card_balance["MONTHS_BALANCE"].replace(0, np.nan)
    credit_card_balance["DAYS_PAST_DUE_DEF_RATIO"] = credit_card_balance[
        "SK_DPD_DEF"
    ] / credit_card_balance["MONTHS_BALANCE"].replace(0, np.nan)
    all_time_credit_card_balance_aggregated = min_max_mean_mode_aggregation(
        credit_card_balance, "SK_ID_CURR", ["NAME_CONTRACT_STATUS"]
    )
    recent_credit_card_balance_aggregated = mean_aggregation_of_recent(
        credit_card_balance, "SK_ID_CURR", "MONTHS_BALANCE", 5
    )
    credit_card_balance_aggregated = all_time_credit_card_balance_aggregated.merge(
        recent_credit_card_balance_aggregated, on="SK_ID_CURR", how="left"
    )
    credit_card_balance_aggregated = credit_card_balance_aggregated.rename(
        columns={
            col: f"CARD_{col}"
            for col in credit_card_balance_aggregated.columns
            if col != "SK_ID_CURR"
        }
    )
    credit_card_balance_aggregated.drop(
        columns=["CARD_SK_DPD_MIN", "CARD_SK_DPD_DEF_MIN"], inplace=True
    )

    # Final features
    loans = application.merge(
        previous_application_aggregated, on="SK_ID_CURR", how="left"
    )
    loans = loans.merge(bureau_merged_aggregated, on="SK_ID_CURR", how="left")
    loans = loans.merge(pos_cash_balance_aggregated, on="SK_ID_CURR", how="left")
    loans = loans.merge(installments_payments_aggregated, on="SK_ID_CURR", how="left")
    loans = loans.merge(credit_card_balance_aggregated, on="SK_ID_CURR", how="left")
    loans = loans.drop(columns=["SK_ID_CURR"] + unimportant_features, errors="ignore")
    loans = downcasting(loans)
    if "TARGET" in loans:
        X = loans.drop(columns=["TARGET"])
        y = loans["TARGET"]
        y_encoded = LabelEncoder().fit_transform(y)
        return X, y_encoded
    else:
        return loans


unimportant_iter_one = [
    "APPLICATION_TO_GOODS_PRICE_RATIO_MIN",
    "CARD_CNT_DRAWINGS_POS_CURRENT_MIN",
    "NFLAG_LAST_APPL_IN_DAY_MEAN_CONSUMER_LOANS",
    "CARD_CNT_DRAWINGS_CURRENT_MIN",
    "CARD_CNT_DRAWINGS_ATM_CURRENT_MIN",
    "CARD_AMT_DRAWINGS_CURRENT_MIN",
    "CARD_AMT_DRAWINGS_ATM_CURRENT_MIN",
    "MONTHLY_DELAY_RATIO_MAX",
    "AVG_DPD_PER_INSTALMENT_MIN",
    "DPD_TO_REMAINING_INSTALMENTS_RATIO_MIN",
    "SK_DPD_MIN",
    "CARD_RECENT_AMT_DRAWINGS_OTHER_CURRENT_MEAN",
    "FIVE_MIN",
    "FOUR_MIN",
    "FOUR_MAX",
    "OVERDUE_DAYS_RATIO_MAX",
    "FLAG_LAST_APPL_PER_CONTRACT_MODE",
    "CREDIT_DAY_OVERDUE_MIN",
    "NFLAG_LAST_APPL_IN_DAY_MIN",
    "DAYS_FIRST_DRAWING_MAX",
    "CARD_OTHER_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO_MIN",
    "CARD_RECENT_SK_DPD_DEF_MEAN",
    "FOURE_MEAN",
    "CARD_RECENT_CNT_DRAWINGS_OTHER_CURRENT_MEAN",
    "THREE_MEAN",
    "FOUR_MAX",
    "OVERDUE_DAYS_RATIO_MIN",
    "CARD_RECENT_DAYS_PAST_DUE_DEF_RATIO_MEAN",
    "CARD_AMT_PAYMENT_TOTAL_CURRENT_MIN",
    "OVERDUE_DAYS_RATIO_MEAN",
    "NFLAG_LAST_APPL_IN_DAY_MEAN",
    "FIVE_MEAN",
    "CARD_CNT_DRAWINGS_OTHER_CURRENT_MAX",
    "CARD_CNT_INSTALMENT_MATURE_CUM_MIN",
    "Recent_Status_Mode_MODE",
    "AMT_CREDIT_SUM_OVERDUE_MEAN",
    "TWO_MEAN",
    "FLAG_DOCUMENT_6",
    "CARD_ATM_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO_MAX",
    "CARD_ATM_DRAWINGS_COUNT_RATIO_MIN",
    "CARD_POS_DRAWINGS_COUNT_RATIO_MAX",
    "CARD_ATM_DRAWINGS_COUNT_RATIO_MAX",
    "CARD_RECEIVABLE_TO_CREDIT_LIMIT_RATIO_MIN",
    "CARD_POS_DRAWINGS_COUNT_RATIO_MIN",
    "TWO_MAX",
    "CARD_BALANCE_TO_CREDIT_LIMIT_RATIO_MIN",
    "NFLAG_INSURED_ON_APPROVAL_MAX",
    "MAX_STATUS_MAX",
    "CARD_RECENT_AMT_TOTAL_RECEIVABLE_MEAN",
    "ONE_MIN",
    "CARD_AMT_BALANCE_MIN",
    "CREDIT_DAY_OVERDUE_MEAN",
    "CARD_ATM_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO_MIN",
    "CARD_POS_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO_MAX",
    "CARD_AMT_TOTAL_RECEIVABLE_MEAN",
    "CARD_AMT_RECEIVABLE_PRINCIPAL_MIN",
    "FIVE_MAX",
    "CARD_DAYS_PAST_DUE_DEF_RATIO_MEAN",
]

unimportant_iter_three = [
    "FLAG_DOCUMENT_14",
    "REG_REGION_NOT_WORK_REGION",
    "EMERGENCYSTATE_MODE",
    "FLAG_DOCUMENT_13",
    "FLAG_OWN_CAR",
    "CARD_AMT_CREDIT_LIMIT_ACTUAL_MAX",
    "CARD_ATM_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO_MEAN",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "CARD_AMT_TOTAL_RECEIVABLE_MIN",
    "CARD_RECENT_AMT_RECEIVABLE_PRINCIPAL_MEAN",
    "CARD_AMT_BALANCE_MEAN",
    "FLAG_DOCUMENT_8",
    "AVG_DPD_DEF_PER_INSTALMENT_MAX",
    "CHILDREN_PER_FAMILY_MEMBER",
    "RECENT_AVG_DPD_PER_INSTALMENT_MEAN",
    "CARD_AMT_PAYMENT_TOTAL_CURRENT_MAX",
    "CARD_CNT_DRAWINGS_ATM_CURRENT_MAX",
    "DPD_DEF_TO_DPD_RATIO_MIN",
    "CARD_RECENT_ATM_DRAWINGS_COUNT_RATIO_MEAN",
    "WALLSMATERIAL_MODE",
    "AMT_GOODS_PRICE_MEAN_REVOLVING_LOANS",
    "AMT_CREDIT_SUM_LIMIT_MIN",
    "CARD_AMT_DRAWINGS_CURRENT_MAX",
    "NAME_HOUSING_TYPE",
    "NFLAG_INSURED_ON_APPROVAL_MEAN_CASH_LOANS",
    "SK_DPD_MEAN",
    "CARD_AMT_PAYMENT_CURRENT_MAX",
    "CARD_MONTHS_BALANCE_MIN",
    "DOWN_PAYMENT_TO_APPLICATION_RATIO_MEAN_CONSUMER_LOANS",
    "ELEVATORS_MODE",
    "CARD_AMT_BALANCE_MAX",
    "CARD_AMT_CREDIT_LIMIT_ACTUAL_MIN",
    "CARD_MONTHS_BALANCE_MEAN",
    "CARD_RECENT_AMT_DRAWINGS_POS_CURRENT_MEAN",
    "NFLAG_INSURED_ON_APPROVAL_MEAN_CONSUMER_LOANS",
    "CARD_PAYMENT_TOTAL_TO_TOTAL_RECEIVABLE_RATIO_MEAN",
    "CREDIT_TO_APPLICATION_RATIO_MEAN_REVOLVING_LOANS",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "RECENT_DPD_TO_REMAINING_INSTALMENTS_RATIO_MEAN",
    "AVG_DPD_PER_INSTALMENT_MAX",
    "NAME_CASH_LOAN_PURPOSE_MODE",
    "CARD_AMT_PAYMENT_TOTAL_CURRENT_MEAN",
    "DPD_TO_REMAINING_INSTALMENTS_RATIO_MAX",
    "X_MAX",
    "CNT_CHILDREN",
    "CARD_AMT_RECIVABLE_MIN",
    "CARD_PAYMENT_TOTAL_TO_TOTAL_RECEIVABLE_RATIO_MAX",
    "DOWN_PAYMENT_TO_APPLICATION_RATIO_MIN",
]

unimportant_iter_two = [
    "MAX_STATUS_MIN",
    "NAME_CLIENT_TYPE_MODE",
    "CARD_RECENT_AMT_INST_MIN_REGULARITY_MEAN",
    "CARD_MONTHS_BALANCE_MAX",
    "CARD_CNT_DRAWINGS_OTHER_CURRENT_MEAN",
    "CNT_INSTALMENT_FUTURE_MIN",
    "CARD_RECENT_OTHER_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO_MEAN",
    "CARD_DAYS_PAST_DUE_RATIO_MIN",
    "NAME_PRODUCT_TYPE_MODE",
    "CARD_POS_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO_MIN",
    "DAYS_FIRST_DRAWING_MIN",
    "CARD_AMT_RECIVABLE_MEAN",
    "CARD_SK_DPD_MEAN",
    "CARD_AMT_RECEIVABLE_PRINCIPAL_MAX",
    "CARD_ATM_DRAWINGS_COUNT_RATIO_MEAN",
    "CARD_RECENT_AMT_RECIVABLE_MEAN",
    "DAYS_TERMINATION_MEAN_REVOLVING_LOANS",
    "NFLAG_INSURED_ON_APPROVAL_MIN",
    "CARD_AMT_TOTAL_RECEIVABLE_MAX",
    "LIVE_CITY_NOT_WORK_CITY",
    "CARD_RECENT_CNT_DRAWINGS_POS_CURRENT_MEAN",
    "CNT_CREDIT_PROLONG_MAX",
    "CARD_AMT_RECEIVABLE_PRINCIPAL_MEAN",
    "CARD_RECENT_POS_DRAWINGS_COUNT_RATIO_MEAN",
    "CARD_POS_DRAWINGS_COUNT_RATIO_MEAN",
    "CARD_RECENT_POS_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO_MEAN",
    "CARD_DAYS_PAST_DUE_RATIO_MEAN",
    "C_MIN",
    "CARD_RECENT_AMT_DRAWINGS_CURRENT_MEAN",
    "NAME_PAYMENT_TYPE_MODE",
    "CARD_OTHER_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO_MAX",
    "ANNUITY_TO_CREDIT_RATIO_MEAN_REVOLVING_LOANS",
    "CARD_POS_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO_MEAN",
    "CNT_PAYMENT_MIN",
    "DPD_DEF_TO_DPD_RATIO_MAX",
    "CARD_AMT_DRAWINGS_ATM_CURRENT_MAX",
    "Last_Status_at_Most_Recent_Month_MODE",
    "CARD_CNT_DRAWINGS_POS_CURRENT_MEAN",
    "SK_ID_PREV_COUNT_REVOLVING_LOANS",
    "CARD_RECENT_AMT_BALANCE_MEAN",
    "CARD_AMT_INST_MIN_REGULARITY_MEAN",
    "CARD_AMT_RECIVABLE_MAX",
    "DAYS_LAST_DUE_MEAN_REVOLVING_LOANS",
    "Consecutive_Zeros_Before_C_MIN",
    "AVG_DPD_PER_INSTALMENT_MEAN",
    "CARD_TOTAL_RECEIVABLE_TO_CREDIT_LIMIT_RATIO_MIN",
    "CARD_OTHER_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO_MEAN",
    "CARD_AMT_INST_MIN_REGULARITY_MAX",
    "INSTALMENTS_COMPLETION_RATIO_MAX",
]

unimportant_iter_four = [
    "NAME_CONTRACT_TYPE_MODE",
    "CONSECUTIVE_ZEROS_BEFORE_C_MAX",
    "CARD_AMT_DRAWINGS_ATM_CURRENT_MEAN",
    "HOUR_APPR_PROCESS_START_MAX",
    "AMT_ANNUITY_MEAN_REVOLVING_LOANS",
    "HOUR_APPR_PROCESS_START_MEAN_REVOLVING_LOANS",
    "ZERO_MAX",
    "FLOORSMIN_AVG",
    "FIRST_DRAWING_TO_TERMINATION_RATIO_MEAN_CASH_LOANS",
    "CARD_RECENT_ATM_DRAWINGS_TO_TOTAL_DRAWINGS_RATIO_MEAN",
    "RECENT_SK_DPD_MEAN",
    "AMT_CREDIT_MEAN_CASH_LOANS",
    "DAYS_LAST_DUE_MEAN",
    "CREDIT_TO_APPLICATION_RATIO_MEAN_CASH_LOANS",
    "DOWN_PAYMENT_TO_GOODS_PRICE_RATIO_MIN",
    "DAYS_FIRST_DRAWING_MEAN_REVOLVING_LOANS",
    "DOWN_PAYMENT_TO_APPLICATION_RATIO_MAX",
    "RATE_DOWN_PAYMENT_MEAN_CONSUMER_LOANS",
    "ONE_MEAN",
    "REG_CITY_NOT_WORK_CITY",
    "AMT_ANNUITY_MEAN_x",
    "DAYS_TERMINATION_MEAN_CONSUMER_LOANS",
    "AMT_GOODS_PRICE_MEAN_CASH_LOANS",
    "DOWN_PAYMENT_TO_GOODS_PRICE_RATIO_MEAN_CONSUMER_LOANS",
    "CREDIT_ACTIVE_MODE",
    "C_MEAN",
    "AMT_CREDIT_MEAN_REVOLVING_LOANS",
    "AMT_CREDIT_SUM_OVERDUE_MAX",
    "NAME_TYPE_SUITE_MODE",
    "CARD_PAYMENT_TO_RECEIVABLE_RATIO_MEAN",
    "NFLAG_INSURED_ON_APPROVAL_MEAN",
    "LIVINGAREA_MODE",
    "INSTALLMENTS_TO_CREDIT_RATIO_MAX",
    "AMT_APPLICATION_MEAN_CASH_LOANS",
    "ON_TIME_INSTALMENTS_RATIO_MIN",
    "CARD_AMT_DRAWINGS_POS_CURRENT_MAX",
    "DOWN_PAYMENT_TO_CREDIT_RATIO_MAX",
    "DOWN_PAYMENT_TO_APPLICATION_RATIO_MEAN",
    "PAYMENT_TO_GOODS_PRICE_RATIO_MEAN_CASH_LOANS",
    "LIVINGAPARTMENTS_AVG",
    "ANNUITY_TO_CREDIT_RATIO_MIN_x",
    "CNT_PAYMENT_MEAN_CONSUMER_LOANS",
    "NAME_YIELD_GROUP_MODE",
    "DAYS_FIRST_DUE_MEAN_REVOLVING_LOANS",
    "MONTHLY_DELAY_RATIO_MIN",
    "CNT_INSTALMENT_FUTURE_MAX",
    "PAYMENT_TO_GOODS_PRICE_RATIO_MAX",
    "YEARS_BEGINEXPLUATATION_MODE",
    "AMT_APPLICATION_MIN",
]

unimportant_iter_five = [
    "CARD_AMT_DRAWINGS_CURRENT_MEAN",
    "CARD_CNT_DRAWINGS_POS_CURRENT_MAX",
    "FLAG_DOCUMENT_18",
    "NAME_PORTFOLIO_MODE",
    "CREDIT_LIMIT_RATIO_MIN",
    "ELEVATORS_AVG",
    "INSTALLMENTS_TO_CREDIT_RATIO_MIN",
    "DPD_TO_REMAINING_INSTALMENTS_RATIO_MEAN",
    "CREDIT_TO_APPLICATION_RATIO_MEAN_CONSUMER_LOANS",
    "GOODS_PRICE_TO_CREDIT_RATIO_MEAN_CONSUMER_LOANS",
    "CHANNEL_TYPE_MODE",
    "DAYS_FIRST_DUE_MAX",
    "CARD_BALANCE_TO_CREDIT_LIMIT_RATIO_MEAN",
    "FLAG_DOCUMENT_16",
    "CARD_CNT_INSTALMENT_MATURE_CUM_MAX",
    "CARD_RECEIVABLE_TO_CREDIT_LIMIT_RATIO_MEAN",
    "DAYS_FIRST_DUE_MEAN_CONSUMER_LOANS",
    "SK_ID_PREV_COUNT_CASH_LOANS",
    "AMT_CREDIT_MEAN",
    "NONLIVINGAREA_MODE",
    "CARD_PAYMENT_TO_RECEIVABLE_RATIO_MAX",
    "RATE_DOWN_PAYMENT_MEAN",
    "TERMINATION_TO_LAST_DUE_RATIO_MEAN_CASH_LOANS",
    "CARD_CNT_DRAWINGS_CURRENT_MAX",
    "HOUR_APPR_PROCESS_START_MIN",
    "AMT_GOODS_PRICE_MEAN",
    "AMT_GOODS_PRICE_MAX",
    "DAYS_TERMINATION_MEAN",
    "ANNUITY_TO_CREDIT_RATIO_MEAN_CONSUMER_LOANS",
    "WEEKDAY_APPR_PROCESS_START_MODE",
    "Max_Status_MEAN",
    "AMT_ANNUITY_MEAN_CASH_LOANS",
    "DAYS_FIRST_DRAWING_MEAN",
    "CNT_INSTALMENT_MEAN",
    "SELLERPLACE_AREA_MEAN_REVOLVING_LOANS",
    "TOTALAREA_MODE",
    "CNT_INSTALMENT_MAX",
    "TOTAL_LATE_PAYMENTS",
    "CARD_RECENT_AMT_PAYMENT_TOTAL_CURRENT_MEAN",
]

unimportant_iter_six = [
    "NAME_CONTRACT_STATUS_MODE_y",
    "APPLICATION_TO_GOODS_PRICE_RATIO_MEAN",
    "CNT_CREDIT_PROLONG_MIN",
    "PROLONG_TO_CREDIT_RATIO_MAX",
    "NFLAG_LAST_APPL_IN_DAY_MEAN_CASH_LOANS",
    "INSTALMENTS_COMPLETION_RATIO_MIN",
    "CNT_CREDIT_PROLONG_MEAN",
    "CREDIT_DAY_OVERDUE_MAX",
    "CARD_SK_DPD_DEF_MEAN",
    "CARD_AMT_DRAWINGS_OTHER_CURRENT_MAX",
    "RECENT_DPD_DEF_TO_DPD_RATIO_MEAN",
    "SK_DPD_MAX",
    "SK_DPD_DEF_MAX",
    "AMT_CREDIT_MAX_OVERDUE_MIN",
]

unimportant_iter_seven = [
    "TWO_MIN",
    "SK_DPD_DEF_MIN",
    "ON_TIME_INSTALMENTS_RATIO_MAX",
    "NFLAG_INSURED_ON_APPROVAL_MEAN_REVOLVING_LOANS",
    "CARD_SK_DPD_MAX",
]

unimportant_iter_eight = [
    "CARD_AMT_PAYMENT_CURRENT_MIN",
    "NFLAG_LAST_APPL_IN_DAY_MEAN_REVOLVING_LOANS",
    "FLAG_PHONE",
    "CARD_AMT_DRAWINGS_POS_CURRENT_MEAN",
    "ON_TIME_INSTALMENTS_RATIO_MEAN",
]

unimportant_iter_nine = [
    "THREE_MAX",
    "FOUR_MEAN",
    "CONSECUTIVE_ZEROS_BEFORE_C_MIN",
    "RECENT_STATUS_MODE_MODE",
    "CNT_INSTALMENT_MIN",
]

unimportant_features = (
    unimportant_iter_one
    + unimportant_iter_two
    + unimportant_iter_three
    + unimportant_iter_four
    + unimportant_iter_five
    + unimportant_iter_six
    + unimportant_iter_seven
    + unimportant_iter_eight
    + unimportant_iter_nine
)
