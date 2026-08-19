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
    from db import (
        init_db,
        save_prediction_record,
        save_goal_record,
        fetch_prediction_history,
        fetch_goal_history,
        fetch_student_progress_timeline,
        delete_prediction_record,
        clear_all_records,
        get_database_stats
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
    from src.db import (
        init_db,
        save_prediction_record,
        save_goal_record,
        fetch_prediction_history,
        fetch_goal_history,
        fetch_student_progress_timeline,
        delete_prediction_record,
        clear_all_records,
        get_database_stats
    )
