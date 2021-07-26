using Gadfly: Data, DataFrame, ismissing
using CSV, DataFrames, DataFramesMeta, Statistics, Gadfly

cd(@__DIR__)
include("../utilities.jl")

df = CSV.read("../../data/analytical/cows-analytic.csv", DataFrame)

# -------------------- Yield -----------------------
df.yieldmax = minmax_yield(df; min = false)
df.yieldmin = minmax_yield(df; min = true)
df.yieldrange = df.yieldmax - df.yieldmin

# ---------------- Conductivity -----------------------
df.condmax = minmax_cond(df; min = false)
df.condmin = minmax_cond(df; min = true)
df.condrange = df.condmax - df.condmin

# CSV.write("../../data/analytical/cows-analytic.csv", df)

plot(df,
  layer(x = :yieldmax, Geom.density, color=[colorant"red"]),
  layer(x = :yieldmin, Geom.density, color=[colorant"blue"])
)

df2 = @where(df, :mdi .!= ismissing(:mdi))

plot(df2, 
  x = :mdi, y = :condmax,
  Geom.boxplot
)

plot(df2,
  x = :mdi, y = :interval,
  Geom.boxplot,
  Guide.xticks(ticks=[1,1.4,2,3,4,5,10])
)