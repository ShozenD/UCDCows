# ========== Plot the total yield by day for each cow ======================================

# ===== Import Packages =====
using DataFrames,
      DataFramesMeta,
      CSV,
      GLM,
      Plots

include("utils.jl")

# ===== Import Data =====
# Move to directory of current file
cd(@__DIR__)
# Read file
path = "../../../data/analytical/cows-analytic.csv"
df = CSV.read(path, DataFrame)
df[!, :id] = categorical(df.id)
df[!, :lactnum] = categorical(df.lactnum)

# ===== Data Processing =====
healthy_data, sick_data, _ = splitbyperiod(df, 5, :sick)

# ===== Plot yield vs days in milk for each lactation =====
# Select lactation number
lactnum = 4
# Scatter plot of true yield for cows in the same lactation
plot_yield_byhealth(healthy_data, sick_data, lactnum)