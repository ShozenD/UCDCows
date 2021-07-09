using Gadfly: passmissing, ismissing
using CSV, DataFrames, Gadfly, Statistics, DataFramesMeta, Impute, Dates

include("../visualization.jl")
include("../utilities.jl")

df = CSV.read("data/analytical/cows-analytic.csv", DataFrame)

df.mdurS = Second.(df.tend - df.tbegin)
df.mdurM = round.(df.mdurS, Dates.Minute) .|> Dates.value
df.mdurS = Dates.value.(df.mdurS)

# Normalize Yield with milking time
convert(Vector, df.flowrr) # Flow is not yield / duration

df.ypmrr = df.yieldrr ./ df.mdurS * 60
df.ypmrf = df.yieldrf ./ df.mdurS * 60
df.ypmlr = df.yieldlr ./ df.mdurS * 60
df.ypmlf = df.yieldlf ./ df.mdurS * 60

df2 = @where(df, :ypmrr .!= ismissing(:ypmrr))

plot(df2,
  x = :ypmlf,
  Geom.histogram
)

ismissing.(df.ypmrr) |> sum
ismissing.(df.ypmrf) |> sum
ismissing.(df.ypmlr) |> sum
ismissing.(df.ypmlf) |> sum
