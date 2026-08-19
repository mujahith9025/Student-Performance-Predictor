import os
import sys

# Ensure src and its parent are on path
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from data_loader import (
        load_student_data,
        load_subject_data,
        engineer_features,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        SUBJECT_METADATA
    )
except ImportError:
    from src.data_loader import (
        load_student_data,
        load_subject_data,
        engineer_features,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        SUBJECT_METADATA
    )
