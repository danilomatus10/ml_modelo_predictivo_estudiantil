import pytest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Marital status': [1, 2, 1],
        'Previous qualification (grade)': [150, np.nan, 130],
        'Target': ['Dropout', 'Graduate', 'Enrolled']
    })

def test_preprocessing(sample_data):
    preprocessor = DataPreprocessor()
    transformed = preprocessor.fit_transform(sample_data)
    
    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()