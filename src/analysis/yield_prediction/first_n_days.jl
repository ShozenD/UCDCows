# ========== Analysis with cows where first N days are recorded ============================

# ===== Import Packages =====
using DataFrames,
      DataFramesMeta,
      CSV

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
filtered, summary = filter_cows(df)