using Plots: png
using CSV, DataFrames, DataFramesMeta, Statistics, Gadfly
import Cairo, Fontconfig

cd(@__DIR__)
include("../utilities.jl")
include("../visualization.jl")

df = CSV.read("../../data/analytical/cows-analytic.csv", DataFrame)

function plot_dist(df::DataFrame, 
                   x::Symbol, 
                   label::String, 
                   fn::String)
  p = plot(df,
    x = x, 
    Geom.histogram,
    Guide.xlabel(label)
  )
  draw(PNG(fn), p)
end

plot_dist(df, :dinmilk, "Days in Milk", "../../resources/figures/dist-dinmilk.png")

# -------------------------- Conductivity ---------------------------
plot_dist(df, 
  :condlf, 
  "Conductivity (Left Front)", 
  "../../resources/figures/dist-condlf.png"
)

plot_dist(df, 
  :condlr, 
  "Conductivity (Left Rear)", 
  "../../resources/figures/dist-condlr.png"
)

plot_dist(df, 
  :condrf, 
  "Conductivity (Right Front)", 
  "../../resources/figures/dist-condrf.png"
)

plot_dist(df, 
  :condrr, 
  "Conductivity (Right Rear)", 
  "../../resources/figures/dist-condrr.png"
)

# ------------------------------ Time -------------------------------
plot_dist(df, :interval, "Interval", "../../resources/figures/dist-interval.png")
plot_dist(df, :lastmilkint, "Last Milking Interval", "../../resources/dist-lastmilkint.png")