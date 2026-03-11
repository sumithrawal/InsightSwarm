import os
from langchain.tools import tool

@tool("Data Ingestion Tool")
def ingest_data_tool(filepath: str) -> str:
    """
    Load a CSV or XLSX file and detect column types.
    Returns the dataset profile as a JSON string.
    """
    from .loader import loadFile, detectColumnTypes, getDatasetProfile
    try:
        df = loadFile(filepath)
        colTypes = detectColumnTypes(df)
        profile = getDatasetProfile(df, filepath)
        return f"Successfully loaded {filepath}. Profile: {profile}"
    except Exception as e:
        return f"Error loading file {filepath}: {str(e)}"

@tool("EDA Analyzer Tool")
def analyze_data_tool(filepath: str, target: str = None) -> str:
    """
    Run full Exploratory Data Analysis (EDA) on a dataset.
    Generates charts and stats, and saves them to the 'outputs' directory.
    Target column is optional.
    Returns the paths to the generated reports.
    """
    from .loader import loadFile, detectColumnTypes, suggestTarget
    from .preprocessor import Preprocessor
    from .analyzer import Analyzer
    try:
        df = loadFile(filepath)
        colTypes = detectColumnTypes(df)
        if target is None or target == "":
            target = suggestTarget(df, colTypes)

        preprocessor = Preprocessor(colTypes)
        df_clean = preprocessor._parseDatetimes(df)

        analyzer = Analyzer(df_clean, colTypes, target_col=target)
        report = analyzer.runFullEda()
        return f"EDA completed successfully. Target used: {target}. Check 'outputs/' directory for charts. Report summary: {list(report.keys())}"
    except Exception as e:
        return f"Error running EDA: {str(e)}"

@tool("Model Trainer Tool")
def train_model_tool(filepath: str, target: str) -> str:
    """
    Train and evaluate multiple machine learning models on the dataset to predict the target column.
    Saves the best model to the 'models' directory.
    Requires a target column.
    Returns the best model name and its score.
    """
    from .loader import loadFile, detectColumnTypes
    from .predictor import Predictor
    try:
        df = loadFile(filepath)
        colTypes = detectColumnTypes(df)
        
        predictor = Predictor(df, colTypes, target)
        results = predictor.run()
        
        if not results:
            return "Training failed or no valid target/features found."
            
        best_name = predictor.bestName
        score_key = "test_r2" if predictor.task == "regression" else "test_f1"
        best_score = results[best_name].get(score_key)
        
        return f"Training completed. Best model: {best_name} with {score_key.upper()} = {best_score}. Saved to 'models/'."
    except Exception as e:
        return f"Error training model: {str(e)}"
