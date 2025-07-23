import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
import time

# Define models to compare
models_to_compare = [
    "amazon/chronos-t5-tiny",
    "amazon/chronos-t5-mini",
    "amazon/chronos-t5-small",
    "amazon/chronos-t5-base",
    # "amazon/chronos-t5-large", # big, slow, poor performance
]

# Load ABT data and filter for Pennsylvania
df_all = pd.read_csv("ABT.csv")
print("Original ABT data shape:", df_all.shape)

# Filter for PA and sort by epiweek
df = df_all[df_all["state"] == "PA"].copy()
df = df.sort_values("epiweek").reset_index(drop=True)

print("PA data shape:", df.shape)
print("First few rows:")
print(df[["epiweek", "hosp-value", "hosp-weekly_rate"]].head())

# Check data patterns
print(f"\nPA Hospital Data Statistics:")
print(f"Mean: {df['hosp-value'].mean():.1f}")
print(f"Std: {df['hosp-value'].std():.1f}")
print(f"Min: {df['hosp-value'].min()}")
print(f"Max: {df['hosp-value'].max()}")
print(f"Total weeks available: {len(df)}")

# Set forecasting parameters to match Moirai analysis
row_cut = 97  # Use same cutoff as improved Moirai script
prediction_length = len(df) - row_cut  # Forecast until end of data

print(f"\nForecasting Setup:")
print(f"Training weeks: {row_cut} (rows 0 to {row_cut-1})")
print(f"Forecast weeks: {prediction_length} (rows {row_cut} to {len(df)-1})")
print(
    f"Epiweek range - Training: {df.iloc[0]['epiweek']} to {df.iloc[row_cut-1]['epiweek']}"
)
print(
    f"Epiweek range - Testing: {df.iloc[row_cut]['epiweek']} to {df.iloc[-1]['epiweek']}"
)

# Extract true values for comparison
true_vals = df["hosp-value"].iloc[row_cut : row_cut + prediction_length]
df_context = df.iloc[:row_cut]

print(f"\nTraining data stats:")
print(f"Training mean: {df_context['hosp-value'].mean():.1f}")
print(
    f"Training range: {df_context['hosp-value'].min()} to {df_context['hosp-value'].max()}"
)
print(f"Last 5 training values: {df_context['hosp-value'].tail().values}")

print(f"\nTest data stats:")
print(f"Test mean: {true_vals.mean():.1f}")
print(f"Test range: {true_vals.min()} to {true_vals.max()}")
print(f"First 5 test values: {true_vals.head().values}")

# Create context tensor for Chronos
context = torch.tensor(df_context["hosp-value"].values, dtype=torch.float32)

# Storage for results
model_results = {}
model_performance = {}

print("\n" + "=" * 60)
print("STARTING CHRONOS MODEL COMPARISON ON PA HOSPITAL DATA")
print("=" * 60)
print(f"Context length: {len(context)}")
print(f"Prediction length: {prediction_length}")
print("-" * 60)

# Compare each model
for model_name in models_to_compare:
    print(f"Testing {model_name}...")

    # Load model
    start_time = time.time()
    try:
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        load_time = time.time() - start_time
        print(f"  Model loaded in {load_time:.2f}s")

        # Make prediction
        start_time = time.time()
        forecast = pipeline.predict(context, prediction_length)
        inference_time = time.time() - start_time

        # Extract quantiles
        low, median, high = np.quantile(
            forecast[0].numpy(), [0.025, 0.5, 0.975], axis=0
        )

        # Store results
        model_results[model_name] = {
            "low": low,
            "median": median,
            "high": high,
            "load_time": load_time,
            "inference_time": inference_time,
        }

        # Calculate performance metrics
        mae = np.mean(np.abs(median - true_vals.values))
        mse = np.mean((median - true_vals.values) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((median - true_vals.values) / true_vals.values)) * 100

        # Calculate coverage (percentage of true values within prediction interval)
        coverage = np.mean((true_vals.values >= low) & (true_vals.values <= high)) * 100

        model_performance[model_name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Coverage_95%": coverage,
            "Load_Time": load_time,
            "Inference_Time": inference_time,
        }

        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")
        print(
            f"  Coverage: {coverage:.1f}%, Load: {load_time:.2f}s, Inference: {inference_time:.3f}s"
        )

        # Check prediction range
        pred_min, pred_max = median.min(), median.max()
        print(
            f"  Prediction range: {pred_min:.0f} to {pred_max:.0f} (actual: {true_vals.min():.0f} to {true_vals.max():.0f})"
        )

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        continue

    print()

# Create comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Chronos Model Comparison on PA Hospital Data (ABT.csv)", fontsize=16)

# Color palette for different models
colors = ["red", "blue", "green", "orange", "purple"]
forecast_index = range(row_cut, row_cut + prediction_length)

# Plot 1: All median forecasts
ax1 = axes[0, 0]
ax1.plot(
    range(len(df_context)),
    df_context["hosp-value"],
    color="black",
    label="Historical data",
    alpha=0.7,
    linewidth=1,
)
ax1.plot(
    forecast_index,
    true_vals,
    color="darkgreen",
    label="True values",
    linewidth=2,
    marker="o",
    markersize=3,
)

for i, (model_name, results) in enumerate(model_results.items()):
    model_short = model_name.split("/")[-1]  # Get short name
    ax1.plot(
        forecast_index,
        results["median"],
        color=colors[i],
        label=f"{model_short}",
        linewidth=2,
    )

ax1.set_title("Median Forecasts Comparison")
ax1.set_xlabel("Week Index")
ax1.set_ylabel("Hospital Admissions")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Best performing model with prediction intervals
if model_performance:
    best_model = min(
        model_performance.keys(), key=lambda x: model_performance[x]["RMSE"]
    )
    best_results = model_results[best_model]
    best_mae = model_performance[best_model]["MAE"]
    best_mape = model_performance[best_model]["MAPE"]

    ax2 = axes[0, 1]
    ax2.plot(
        range(len(df_context)),
        df_context["hosp-value"],
        color="black",
        label="Historical data",
        alpha=0.7,
        linewidth=1,
    )
    ax2.plot(
        forecast_index,
        true_vals,
        color="darkgreen",
        label="True values",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax2.plot(
        forecast_index,
        best_results["median"],
        color="red",
        label=f"{best_model.split('/')[-1]} median",
        linewidth=2,
    )
    ax2.fill_between(
        forecast_index,
        best_results["low"],
        best_results["high"],
        color="red",
        alpha=0.3,
        label="95% prediction interval",
    )
    ax2.set_title(
        f'Best Model: {best_model.split("/")[-1]} (MAE: {best_mae:.0f}, MAPE: {best_mape:.1f}%)'
    )
    ax2.set_xlabel("Week Index")
    ax2.set_ylabel("Hospital Admissions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Plot 3: Performance metrics comparison
if model_performance:
    ax3 = axes[1, 0]
    model_names = [name.split("/")[-1] for name in model_performance.keys()]
    mae_values = [model_performance[name]["MAE"] for name in model_performance.keys()]
    rmse_values = [model_performance[name]["RMSE"] for name in model_performance.keys()]

    x = np.arange(len(model_names))
    width = 0.35

    ax3.bar(x - width / 2, mae_values, width, label="MAE", alpha=0.7, color="skyblue")
    ax3.bar(x + width / 2, rmse_values, width, label="RMSE", alpha=0.7, color="orange")

    ax3.set_xlabel("Models")
    ax3.set_ylabel("Error")
    ax3.set_title("Error Metrics Comparison")
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Timing comparison
    ax4 = axes[1, 1]
    load_times = [
        model_performance[name]["Load_Time"] for name in model_performance.keys()
    ]
    inference_times = [
        model_performance[name]["Inference_Time"] for name in model_performance.keys()
    ]

    ax4.bar(
        x - width / 2,
        load_times,
        width,
        label="Load Time",
        alpha=0.7,
        color="lightcoral",
    )
    ax4.bar(
        x + width / 2,
        inference_times,
        width,
        label="Inference Time",
        alpha=0.7,
        color="lightgreen",
    )

    ax4.set_xlabel("Models")
    ax4.set_ylabel("Time (seconds)")
    ax4.set_title("Performance Timing Comparison")
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print detailed performance comparison
if model_performance:
    print("\n" + "=" * 80)
    print("DETAILED PERFORMANCE COMPARISON")
    print("=" * 80)

    # Create performance DataFrame for easy comparison
    perf_df = pd.DataFrame(model_performance).T
    perf_df.index = [name.split("/")[-1] for name in perf_df.index]

    print("\nPerformance Metrics:")
    print(perf_df.round(3))

    print(f"\nBest Models by Metric:")
    print(f"Lowest MAE: {perf_df['MAE'].idxmin()} ({perf_df['MAE'].min():.2f})")
    print(f"Lowest RMSE: {perf_df['RMSE'].idxmin()} ({perf_df['RMSE'].min():.2f})")
    print(f"Lowest MAPE: {perf_df['MAPE'].idxmin()} ({perf_df['MAPE'].min():.1f}%)")
    print(
        f"Best Coverage: {perf_df['Coverage_95%'].idxmax()} ({perf_df['Coverage_95%'].max():.1f}%)"
    )
    print(
        f"Fastest Load: {perf_df['Load_Time'].idxmin()} ({perf_df['Load_Time'].min():.2f}s)"
    )
    print(
        f"Fastest Inference: {perf_df['Inference_Time'].idxmin()} ({perf_df['Inference_Time'].min():.3f}s)"
    )

    print("\n" + "=" * 80)
    print("SUMMARY & ANALYSIS")
    print("=" * 80)
    print(f"Overall best model (lowest RMSE): {perf_df['RMSE'].idxmin()}")
    print(
        f"Best balance of accuracy and speed: {perf_df['MAE'].idxmin()} (considering MAE)"
    )

    print(f"\nData Challenge Analysis:")
    print(f"Training data ended at: {df_context['hosp-value'].tail(5).values}")
    print(f"Test data started at: {true_vals.head(5).values}")
    print(f"Seasonal surge pattern: {true_vals.min():.0f} → {true_vals.max():.0f}")

    print("\nModel size vs performance trade-offs:")
    for model in perf_df.index:
        mae_val = perf_df.loc[model, "MAE"]
        coverage_val = perf_df.loc[model, "Coverage_95%"]
        inference_val = perf_df.loc[model, "Inference_Time"]
        print(
            f"  {model:15s}: MAE={mae_val:6.1f}, Coverage={coverage_val:5.1f}%, Inference={inference_val:6.3f}s"
        )

    # Compare with typical baselines
    print(f"\nBaseline Comparisons:")
    naive_forecast = np.full(
        prediction_length, df_context["hosp-value"].iloc[-1]
    )  # Last value
    naive_mae = np.mean(np.abs(naive_forecast - true_vals.values))

    seasonal_naive = (
        []
    )  # Simple seasonal naive (use value from 52 weeks ago if available)
    for i in range(prediction_length):
        look_back_idx = len(df_context) - 52 + i
        if look_back_idx >= 0 and look_back_idx < len(df_context):
            seasonal_naive.append(df_context["hosp-value"].iloc[look_back_idx])
        else:
            seasonal_naive.append(df_context["hosp-value"].iloc[-1])

    seasonal_naive = np.array(seasonal_naive)
    seasonal_mae = np.mean(np.abs(seasonal_naive - true_vals.values))

    print(f"Naive (last value): MAE = {naive_mae:.2f}")
    print(f"Seasonal naive: MAE = {seasonal_mae:.2f}")

    if len(perf_df) > 0:
        best_chronos_mae = perf_df["MAE"].min()
        print(f"Best Chronos model: MAE = {best_chronos_mae:.2f}")
        print(
            f"Improvement vs naive: {((naive_mae - best_chronos_mae) / naive_mae * 100):+.1f}%"
        )
        print(
            f"Improvement vs seasonal: {((seasonal_mae - best_chronos_mae) / seasonal_mae * 100):+.1f}%"
        )

else:
    print("No models completed successfully!")
