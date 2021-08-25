using CSV: makeunique
using DataFrames: make_unique
using Gadfly: DataFrame

cd(@__DIR__)

### Load Packages and utility functions
using CSV, DataFrames, Gadfly, Statistics, DataFramesMeta, Impute

include("../visualization.jl")
include("../utilities.jl")

### Read data 
df_cows = CSV.read("../../data/analytical/cows-analytic.csv", DataFrame)
df_diag = CSV.read("../../data/diagnostic/diagnosis-2-clean.csv", DataFrame)

df_diag = @where(df_diag, ismissing.(:dinmilk) .!= 1)

df = leftjoin(df_cows, df_diag, on = [:id, :lactnum], makeunique = true)

df_mst = @where(df, :mastflag .== 1)

unique(df_mst.id)

sort(df.date)
sort(df_diag, :datehosp)

plot_conductivity(df_cows, 43469, 1, 4)
plot_yield(df_cows, 43469, 1.0, 4)
plot_ypm(df_cows, unique(df.id)[11], 1.0, 10)

@where(df_diag, :id .== 43469)

### Conclusion
# Diagnosis information and cow data does not overlap...