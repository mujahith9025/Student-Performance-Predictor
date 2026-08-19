import os
import pytest
import pandas as pd
from src.db import (
    init_db,
    save_prediction_record,
    save_goal_record,
    fetch_prediction_history,
    fetch_goal_history,
    fetch_student_progress_timeline,
    delete_prediction_record,
    get_database_stats
)

def test_database_crud_operations():
    """Verify end-to-end database initialization, insertion, query, and deletion."""
    init_db()
    
    # 1. Insert prediction
    rec_id = save_prediction_record(
        student_name="CI Test Student",
        subject="Science & Physics",
        model_used="XGBoost Regressor",
        hours_studied=7.0,
        previous_score=85.0,
        sleep_hours=8.0,
        sample_papers=5,
        extracurricular="Yes",
        predicted_score=88.5,
        lower_bound=84.5,
        upper_bound=92.5,
        confidence_level="95%",
        letter_grade="A",
        standing_desc="Excellent",
        habit_balance_score=90.0
    )
    assert rec_id is not None
    assert rec_id > 0
    
    # 2. Insert goal
    goal_id = save_goal_record(
        student_name="CI Test Student",
        subject="Science & Physics",
        current_pred=88.5,
        target_score=95.0,
        score_gap=6.5,
        recommended_pathway="Sprint Pathway",
        required_hours=8.0,
        required_papers=7,
        required_sleep=7.5,
        weekly_hours=56.0
    )
    assert goal_id is not None
    assert goal_id > 0
    
    # 3. Query history with filter
    df_history = fetch_prediction_history(student_name="CI Test Student")
    assert len(df_history) >= 1
    assert df_history.iloc[0]["student_name"] == "CI Test Student"
    assert df_history.iloc[0]["letter_grade"] == "A"
    
    # 4. Fetch timeline
    df_timeline = fetch_student_progress_timeline("CI Test Student")
    assert len(df_timeline) >= 1
    assert "predicted_score" in df_timeline.columns
    
    # 5. Database stats
    stats = get_database_stats()
    assert stats["total_predictions"] >= 1
    assert stats["unique_students"] >= 1
    
    # 6. Delete test record
    deleted = delete_prediction_record(rec_id)
    assert deleted is True
