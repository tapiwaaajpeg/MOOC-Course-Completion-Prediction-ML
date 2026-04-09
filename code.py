"""
Project: MOOC Course Completion Prediction
=============================================
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log1p, to_date, datediff
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────
DATA_PATH   = "file:///C:/Users/LENOVO/Downloads/archive1/cs_mitx.csv"
OUTPUT_DIR  = r"C:\Users\LENOVO\Desktop"   # all charts saved here
# ─────────────────────────────────────────────────────────────

import os
def save(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + path)


# ══════════════════════════════════════════════════════════════
# STEP 1 — Start Spark & Load Data  (Unit I)
# ══════════════════════════════════════════════════════════════
spark = SparkSession.builder \
    .appName("MOOC_Completion_Prediction") \
    .getOrCreate()

spark.conf.set("spark.sql.debug.maxToStringFields", 100)
spark.sparkContext.setLogLevel("WARN")

df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

print("=== Schema ===")
df.printSchema()
print(f"Total records loaded: {df.count():,}\n")


# ══════════════════════════════════════════════════════════════
# STEP 2 — Clean Nulls & Filter Invalid Sessions
# ══════════════════════════════════════════════════════════════
df = df.dropna(subset=["Completed_or_Not"])
df = df.filter(col("ndays_act") >= 0)
df = df.filter(col("nevents")   >= 0)

df = df.withColumn("nplay_video",
        when(col("nplay_video") == 197757, None).otherwise(col("nplay_video")))

df = df.fillna({
    "nevents":      0,
    "ndays_act":    0,
    "nplay_video":  0,
    "nchapters":    0,
    "nforum_posts": 0,
    "grade":        "0",
})

df = df.withColumn("grade", col("grade").cast("double")).fillna({"grade": 0.0})

print(f"Records after cleaning: {df.count():,}\n")


# ══════════════════════════════════════════════════════════════
# STEP 3 — Feature Engineering  (Unit VI)
# ══════════════════════════════════════════════════════════════
df = df.withColumn("start_dt", to_date(col("start_time_DI"), "d/M/yy")) \
       .withColumn("last_dt",  to_date(col("last_event_DI"),  "d/M/yy"))

df = df.withColumn("active_span_days",
        when(col("last_dt").isNotNull() & col("start_dt").isNotNull(),
             datediff(col("last_dt"), col("start_dt"))).otherwise(0))

df = df.withColumn("video_per_day",
        when(col("ndays_act") > 0,
             col("nplay_video") / col("ndays_act")).otherwise(0.0))

df = df.withColumn("events_per_day",
        when(col("ndays_act") > 0,
             col("nevents") / col("ndays_act")).otherwise(0.0))

df = df.withColumn("log_nevents",
        log1p(col("nevents").cast("double")))

df = df.withColumn("forum_active",
        when(col("nforum_posts") > 0, 1.0).otherwise(0.0))

df = df.withColumn("gender_bin",
        when(col("gender") == "m", 1.0).otherwise(0.0))

# ── Class weights to handle 97/3 imbalance ───────────────────
total     = df.count()
n_dropout = df.filter(col("Completed_or_Not") == 0).count()
n_comp    = df.filter(col("Completed_or_Not") == 1).count()

weight_dropout = total / (2.0 * n_dropout)
weight_comp    = total / (2.0 * n_comp)

print(f"Class weights — Dropout: {weight_dropout:.3f}, Completed: {weight_comp:.3f}\n")

df = df.withColumn("classWeight",
        when(col("Completed_or_Not") == 1, weight_comp).otherwise(weight_dropout))

# grade, explored, normalized_Pa removed — likely label leakage
FEATURE_COLS = [
    "viewed",
    "ndays_act",
    "nchapters",
    "nforum_posts",
    "e_value",
    "v_value",
    "active_span_days",
    "video_per_day",
    "events_per_day",
    "log_nevents",
    "forum_active",
    "gender_bin",
]

assembler = VectorAssembler(
    inputCols=FEATURE_COLS,
    outputCol="features_raw",
    handleInvalid="skip",
)
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,
    withStd=True,
)


# ══════════════════════════════════════════════════════════════
# STEP 4 — Logistic Regression Pipeline  (Unit VI)
# ══════════════════════════════════════════════════════════════
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="Completed_or_Not",
    weightCol="classWeight",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0,
)

pipeline = Pipeline(stages=[assembler, scaler, lr])

print("Training model...")
model       = pipeline.fit(train_df)
predictions = model.transform(test_df)
print("Done.\n")


# ══════════════════════════════════════════════════════════════
# STEP 5 — Evaluate  (Unit VI)
# ══════════════════════════════════════════════════════════════
acc = MulticlassClassificationEvaluator(
    labelCol="Completed_or_Not", predictionCol="prediction", metricName="accuracy"
).evaluate(predictions)

f1 = MulticlassClassificationEvaluator(
    labelCol="Completed_or_Not", predictionCol="prediction", metricName="f1"
).evaluate(predictions)

auc = BinaryClassificationEvaluator(
    labelCol="Completed_or_Not", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
).evaluate(predictions)

print("=== Evaluation Results ===")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  ROC-AUC  : {auc:.4f}\n")


# ══════════════════════════════════════════════════════════════
# STEP 6 — Visualisations
# ══════════════════════════════════════════════════════════════
print("=== Generating Visualisations ===")

C_DROP  = "#FF6B6B"
C_COMP  = "#00C9A7"
C_ACC   = "#4ECDC4"
C_BLUE  = "#45B7D1"

# ── Collect pandas DataFrames once ───────────────────────────
counts_pd = (df.groupBy("Completed_or_Not").count()
               .toPandas().sort_values("Completed_or_Not"))

motivation_pd = (df.groupBy("Motivation_Label", "Completed_or_Not")
                   .count().toPandas())

course_pd = (df.groupBy("course_id", "Completed_or_Not")
               .count().toPandas())

activity_pd = (df.select("ndays_act", "log_nevents",
                          "video_per_day", "active_span_days",
                          "Completed_or_Not")
                 .sample(fraction=0.3, seed=42)
                 .toPandas())

corr_pd = (df.select(FEATURE_COLS + ["Completed_or_Not"])
             .sample(fraction=0.3, seed=42)
             .toPandas())

pred_pd = predictions.select("Completed_or_Not", "prediction",
                              "probability").toPandas()
pred_pd["prob_complete"] = pred_pd["probability"].apply(lambda v: float(v[1]))

lr_model = model.stages[-1]
coefs    = np.abs(lr_model.coefficients.toArray())
feat_imp = sorted(zip(FEATURE_COLS, coefs), key=lambda x: x[1])


# ──────────────────────────────────────────────────────────────
# CHART 1 — Completion vs Dropout Ratio + Motivation Label
#            (original chart, now with imbalance fix applied)
# ──────────────────────────────────────────────────────────────
sizes  = counts_pd["count"].tolist()
labels = ["Dropout", "Completed"]

pivot_mot = motivation_pd.pivot(
    index="Motivation_Label", columns="Completed_or_Not", values="count"
).fillna(0)
pivot_mot.columns = ["Dropout", "Completed"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].pie(sizes, labels=labels, autopct="%1.1f%%",
            colors=[C_DROP, C_COMP], startangle=90,
            wedgeprops=dict(width=0.55))
axes[0].set_title("Completion vs Dropout Ratio")

pivot_mot.plot(kind="bar", ax=axes[1], color=[C_DROP, C_COMP],
               edgecolor="none", width=0.6)
axes[1].set_title("Completion by Motivation Label")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(loc="upper right")

plt.tight_layout()
save("chart1_completion_ratio.png")


# ──────────────────────────────────────────────────────────────
# CHART 2 — Unit I: Spark vs MapReduce Execution Time
# ──────────────────────────────────────────────────────────────
stages_names  = ["Batch Job", "Iterative ML\n(10 iterations)", "Stream\nProcessing"]
mr_times      = [120, 950, 300]
spark_times   = [45,   80,  20]
x             = np.arange(len(stages_names))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 0.2, mr_times,   width=0.38, label="MapReduce", color=C_DROP,  edgecolor="white")
ax.bar(x + 0.2, spark_times, width=0.38, label="Spark",     color=C_COMP, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(stages_names)
ax.set_ylabel("Relative Execution Time (s)")
ax.set_title("Unit I: Spark vs MapReduce — Processing Time Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.3)
for xi, (m, s) in enumerate(zip(mr_times, spark_times)):
    ax.text(xi - 0.2, m + 10, str(m), ha="center", fontsize=9, color=C_DROP)
    ax.text(xi + 0.2, s + 10, str(s), ha="center", fontsize=9, color=C_COMP)

plt.tight_layout()
save("chart2_spark_vs_mapreduce.png")


# ──────────────────────────────────────────────────────────────
# CHART 3 — Unit III: RDD Lineage / DAG
# ──────────────────────────────────────────────────────────────
steps  = ["textFile()\n[Source RDD]", "filter()\n[Transform]",
          "map()\n[Transform]", "reduceByKey()\n[Transform]",
          "collect()\n[Action]"]
colors = [C_ACC, C_BLUE, C_BLUE, C_BLUE, C_DROP]

fig, ax = plt.subplots(figsize=(15, 3.2))
ax.set_xlim(0, 15); ax.set_ylim(0, 2.2); ax.axis("off")
ax.set_title("Unit III: RDD Lineage Graph (DAG) — Transformation Pipeline",
             fontsize=13, fontweight="bold", pad=10)

for i, (label, color) in enumerate(zip(steps, colors)):
    x_pos = 1.3 + i * 2.7
    fancy = mpatches.FancyBboxPatch(
        (x_pos - 1.1, 0.45), 2.1, 1.1,
        boxstyle="round,pad=0.12", facecolor=color, edgecolor="white",
        linewidth=1.5, alpha=0.88,
    )
    ax.add_patch(fancy)
    ax.text(x_pos, 1.0, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
    if i < len(steps) - 1:
        ax.annotate("", xy=(x_pos + 1.65, 1.0), xytext=(x_pos + 1.05, 1.0),
                    arrowprops=dict(arrowstyle="-|>", color="#cccccc",
                                   lw=2.0, mutation_scale=18))

ax.text(0.5, 0.05, "Transformations are lazy — executed only when Action is called",
        fontsize=9, color="grey", style="italic")

plt.tight_layout()
save("chart3_rdd_dag.png")


# ──────────────────────────────────────────────────────────────
# CHART 4 — Unit IV: Spark SQL — Completions per Course (GroupBy)
# ──────────────────────────────────────────────────────────────
pivot_course = course_pd.pivot(
    index="course_id", columns="Completed_or_Not", values="count"
).fillna(0)
pivot_course.columns = ["Dropout", "Completed"]

fig, ax = plt.subplots(figsize=(10, 5))
pivot_course.plot(kind="barh", ax=ax, color=[C_DROP, C_COMP],
                  edgecolor="none", width=0.6)
ax.set_title("Unit IV: Spark SQL GroupBy — Learner Counts per Course")
ax.set_xlabel("Number of Learners")
ax.set_ylabel("Course ID")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
save("chart4_sparksql_groupby_course.png")


# ──────────────────────────────────────────────────────────────
# CHART 5 — Unit V: Simulated Kafka Streaming Ingestion Rate
# ──────────────────────────────────────────────────────────────
np.random.seed(7)
time_steps   = np.arange(0, 60)
base_rate    = 120
event_stream = (np.random.poisson(lam=base_rate, size=60)
                + np.sin(np.linspace(0, 3 * np.pi, 60)) * 30).clip(min=0)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(time_steps, event_stream, color=C_ACC, linewidth=2.0, zorder=3)
ax.fill_between(time_steps, event_stream, alpha=0.2, color=C_ACC)
ax.axhline(y=base_rate, color=C_DROP, linestyle="--",
           linewidth=1.4, label=f"Avg throughput ({base_rate} events/s)")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Events / second")
ax.set_title("Unit V: Kafka Topic Ingestion Rate — Simulated Spark Streaming Window")
ax.legend()
ax.grid(alpha=0.25)

# Annotate a spike
peak_idx = int(np.argmax(event_stream))
ax.annotate(f"Peak: {int(event_stream[peak_idx])} ev/s",
            xy=(time_steps[peak_idx], event_stream[peak_idx]),
            xytext=(time_steps[peak_idx] + 4, event_stream[peak_idx] + 12),
            arrowprops=dict(arrowstyle="->", color="grey"),
            fontsize=9, color="grey")

plt.tight_layout()
save("chart5_kafka_streaming.png")


# ──────────────────────────────────────────────────────────────
# CHART 6 — Unit VI: Correlation Heatmap
# ──────────────────────────────────────────────────────────────
corr = corr_pd.corr()
cols = list(corr.columns)
n    = len(cols)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(cols, fontsize=8)
ax.set_title("Unit VI: Feature Correlation Heatmap", fontsize=13, fontweight="bold")

for i in range(n):
    for j in range(n):
        val = corr.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7,
                color="white" if abs(val) > 0.5 else "black")

plt.tight_layout()
save("chart6_correlation_heatmap.png")


# ──────────────────────────────────────────────────────────────
# CHART 7 — Unit VI: Confusion Matrix
# ──────────────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pred_pd["Completed_or_Not"], pred_pd["prediction"])

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted\nDropout", "Predicted\nCompleted"])
ax.set_yticklabels(["Actual\nDropout", "Actual\nCompleted"])
ax.set_title("Unit VI: Confusion Matrix", fontsize=13, fontweight="bold")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > cm.max() * 0.6 else "black")

plt.tight_layout()
save("chart7_confusion_matrix.png")


# ──────────────────────────────────────────────────────────────
# CHART 8 — Unit VI: ROC Curve
# ──────────────────────────────────────────────────────────────
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(pred_pd["Completed_or_Not"], pred_pd["prob_complete"])

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=C_COMP, lw=2.2, label=f"Logistic Regression (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.2, label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.15, color=C_COMP)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Unit VI: ROC Curve — Logistic Regression", fontsize=12, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

plt.tight_layout()
save("chart8_roc_curve.png")


# ──────────────────────────────────────────────────────────────
# CHART 9 — Unit VI: Feature Importances (LR Coefficients)
# ──────────────────────────────────────────────────────────────
feat_names = [f for f, _ in feat_imp]
feat_vals  = [v for _, v in feat_imp]

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = [C_COMP if v == max(feat_vals) else C_ACC for v in feat_vals]
ax.barh(feat_names, feat_vals, color=bar_colors, edgecolor="none", height=0.6)
ax.set_xlabel("Absolute Coefficient Value")
ax.set_title("Unit VI: Feature Importances — Logistic Regression Coefficients",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
for i, v in enumerate(feat_vals):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8.5)

plt.tight_layout()
save("chart9_feature_importance.png")


# ──────────────────────────────────────────────────────────────
# CHART 10 — Activity distributions by outcome
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("Unit VI: Activity Distributions — Completed vs Dropout",
             fontsize=13, fontweight="bold")

plot_cols = [
    ("ndays_act",        "Active Days"),
    ("log_nevents",      "Log(Events + 1)"),
    ("video_per_day",    "Videos per Active Day"),
    ("active_span_days", "Course Span (days)"),
]

for ax, (pcol, title) in zip(axes.flatten(), plot_cols):
    for label, color, name in [(0, C_DROP, "Dropout"), (1, C_COMP, "Completed")]:
        data = activity_pd[activity_pd["Completed_or_Not"] == label][pcol].dropna()
        ax.hist(data, bins=40, alpha=0.65, color=color, label=name,
                edgecolor="none", density=True)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

plt.tight_layout()
save("chart10_activity_distributions.png")


# ──────────────────────────────────────────────────────────────
# CHART 11 — Model Metrics Summary Bar
# ──────────────────────────────────────────────────────────────
metrics      = ["Accuracy", "F1-Score\n(weighted)", "ROC-AUC"]
metric_vals  = [acc, f1, auc]
bar_c        = [C_COMP, C_ACC, C_BLUE]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(metrics, metric_vals, color=bar_c, edgecolor="none", width=0.45)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Unit VI: Model Evaluation Summary", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, metric_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
            f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
save("chart11_model_metrics.png")


print("\n=== All done ===")
print(f"11 charts saved to {OUTPUT_DIR}")

spark.stop()
