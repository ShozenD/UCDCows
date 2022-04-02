# ========== Analysis with cows where first N days are recorded ============================
# Diff from v1:
#   - Uses dummy variables and have everything fitted into one linear regression equation
#     instead of having multiple equations
#   

# ===== Import Packages =====
using DataFrames,
      DataFramesMeta,
      CSV,
      CategoricalArrays,
      Dates,
      Statistics,
      Plots,
      GLM,
      LaTeXStrings,
      Random,
      StatsBase,
      AverageShiftedHistograms,
      Distributions

# ========== Functions ==========
include("data_cleaning.jl")
include("model_fit.jl")
include("visualization.jl")

## ========== Workflow ==========
# ----- Import Data -----
@info "Running: Data Cleaning"
# Move to directory of current file
cd(@__DIR__)
# Read milk production data file
path = "../../../data/analytical/cows-analytic.csv"
df = CSV.read(path, DataFrame)
df[!, :id] = categorical(df.id)
df[!, :lactnum] = categorical(df.lactnum)

## ----- Data Processing -----
# Keep data of cows with records of their first 30 days in milk, discard the rest of the
# data.
filtered, summary_data = filter_cows(df)
# -- Remove certain cows that have missing teats --
# Cows: 9064 (LR), 49236 (RF), 7984 (RF), 42130 (LR), and 48695 (LR)
filtered = @subset(filtered, :id .∉ Ref([9064, 49236, 7984, 42130, 48695]))
# Split data into healthy and sick
mdi_threshold₁ = 1.4
mdi_threshold₂ = 1.8
df_healthy₁, df_sick₁ = splitbyhealth(filtered, 30, criterion=:mdi, threshold=mdi_threshold₁)
df_healthy₂, df_sick₂ = splitbyhealth(filtered, 30, criterion=:mdi, threshold=mdi_threshold₂)
# -- Remove "healthy" cows with MDi above threshold at any point of milking -- 
# Only keep fully healthy cows
cows = @subset(df_healthy₁, :mdi .≥ mdi_threshold₁) |> x -> unique(x.id)
df_healthy₁ = @subset(df_healthy₁, :id .∉ Ref(cows))
cows = @subset(df_healthy₂, :mdi .≥ mdi_threshold₂) |> x -> unique(x.id)
df_healthy₂ = @subset(df_healthy₂, :id .∉ Ref(cows))
# Remove "healthy" cows with gaps
df_healthy₁ = removecowswithgaps(df_healthy₁)
df_healthy₂ = removecowswithgaps(df_healthy₂)
# --- Aggregate data ---
df_healthy₁ = aggregate_data(df_healthy₁)       # Data where all cows have MDi<1.4 at all times
df_healthy₂ = aggregate_data(df_healthy₂)       # Data where all cows have MDi<1.8 at all times
df_sick₁ = aggregate_data(df_sick₁)             # Data where cows have MDi≥1.4 at certain times
df_sick₂ = aggregate_data(df_sick₂)             # Data where cows have MDi≥1.8 at certain times

## Plot an example of true vs predicted milk yield for a specific cow in each category.
# Healthy: 8244, 8906, 8329, 8169, 7796
# Mid high: 44923, 45890, 45808, 45440
# High: 45808, 9008, 9007
split_date = Date(2021, 11, 1)
resultsByDate = categorize_and_fit(df_healthy₁, df_sick₁, 1, :mdi, mdi_threshold₁, split_by = :date, split_date = split_date)
resultsByProp = categorize_and_fit(df_healthy₁, df_sick₁, 1, :mdi, mdi_threshold₁, split_by = :proportion, train_size = 0.8, test_size = 0.2)