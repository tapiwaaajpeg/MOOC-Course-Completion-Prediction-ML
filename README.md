# MOOC Course Completion Prediction (PySpark)

Predicting whether a learner will **complete** or **drop out of** an online course using **Apache Spark (PySpark)** and a **Logistic Regression** pipeline, trained on MITx learner activity data. The script also produces 11 charts covering the data, the model results, and a few illustrative Spark concept diagrams.

## Overview

MOOCs suffer from very high dropout rates. This project builds a distributed machine learning pipeline in PySpark that learns from learner engagement signals (active days, events, chapters viewed, forum activity, and more) to predict course completion. Because completions are rare, the model uses **class weighting** to handle the heavy imbalance (roughly 97% dropout vs 3% completed, per the code comments).

## Dataset

- **File:** `cs_mitx.csv` (MITx learner records, one row per learner per course)
- **Target:** `Completed_or_Not` (1 = completed, 0 = dropout)
- **Key columns used:** `viewed`, `ndays_act`, `nevents`, `nplay_video`, `nchapters`, `nforum_posts`, `e_value`, `v_value`, `gender`, `start_time_DI`, `last_event_DI`, `course_id`, `Motivation_Label`

## Pipeline

1. **Load data:** start a Spark session and read the CSV with schema inference.
2. **Clean data:**
   - Drop rows with a missing target
   - Remove invalid rows (negative `ndays_act` or `nevents`)
   - Treat the outlier `nplay_video == 197757` as null
   - Fill remaining nulls with 0 and cast `grade` to a numeric type
3. **Feature engineering:** derive activity-span, per-day, log, and binary features (see below).
4. **Class weights:** compute inverse-frequency weights for the dropout and completed classes.
5. **Train:** 80/20 train/test split (seed 42), then a Spark ML `Pipeline` of `VectorAssembler` → `StandardScaler` → `LogisticRegression`.
6. **Evaluate:** accuracy, weighted F1, and ROC-AUC on the test set.
7. **Visualize:** save 11 charts to the output folder.

## Features

| Feature | Description |

| `viewed`, `ndays_act`, `nchapters`, `nforum_posts`, `e_value`, `v_value` | Raw engagement columns |
| `active_span_days` | Days between first and last recorded event |
| `video_per_day` | Videos played per active day |
| `events_per_day` | Events per active day |
| `log_nevents` | `log(1 + nevents)` to reduce skew |
| `forum_active` | 1 if the learner posted in the forum, else 0 |
| `gender_bin` | 1 if gender is `m`, else 0 |

**Leakage prevention:** `grade`, `explored`, and `normalized_Pa` are deliberately excluded because they are likely to leak the label.

## Model and Evaluation

- **Model:** Logistic Regression (`maxIter=100`, `regParam=0.01`, `elasticNetParam=0.0`) with `classWeight` as the weight column
- **Scaling:** `StandardScaler` with mean centering and unit variance
- **Metrics:** Accuracy, weighted F1-score, ROC-AUC (printed to the console and shown in chart 11)
- **Interpretability:** absolute logistic regression coefficients are used as feature importances (chart 9)

Run the script to see the metrics for your data and split.

## Generated Charts

| # | File | Content |

| 1 | `chart1_completion_ratio.png` | Completion vs dropout ratio, and completion by motivation label |
| 2 | `chart2_spark_vs_mapreduce.png` | Spark vs MapReduce processing time (illustrative) |
| 3 | `chart3_rdd_dag.png` | RDD lineage / DAG diagram (illustrative) |
| 4 | `chart4_sparksql_groupby_course.png` | Learner counts per course (Spark SQL GroupBy) |
| 5 | `chart5_kafka_streaming.png` | Simulated Kafka streaming ingestion rate (illustrative) |
| 6 | `chart6_correlation_heatmap.png` | Feature correlation heatmap |
| 7 | `chart7_confusion_matrix.png` | Confusion matrix on the test set |
| 8 | `chart8_roc_curve.png` | ROC curve with AUC |
| 9 | `chart9_feature_importance.png` | Logistic regression coefficient magnitudes |
| 10 | `chart10_activity_distributions.png` | Activity distributions, completed vs dropout |
| 11 | `chart11_model_metrics.png` | Accuracy, F1, and AUC summary |

## Notes and Limitations

- Charts 2, 3, and 5 are **illustrative**: they use hardcoded or simulated values to demonstrate Spark concepts (Spark vs MapReduce, RDD lineage, Kafka streaming) and are not computed from the dataset.
- Charts 6 and 10 use a 30% random sample of the data for speed.
- Date columns are parsed with the format `d/M/yy`; other formats will produce nulls and a span of 0.
- With severe class imbalance, accuracy alone can be misleading. Look at F1, AUC, and the confusion matrix too.
- Paths in the script are hardcoded for a local Windows machine.

## Future Work

- Try tree-based models (Random Forest, Gradient-Boosted Trees) with Spark ML
- Tune hyperparameters with `CrossValidator`
- Compare class weighting with resampling approaches
- Add real streaming ingestion with Spark Structured Streaming and Kafka

## Disclaimer

This project is for educational purposes.
