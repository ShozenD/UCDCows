using Plots: png
using CSV, DataFrames, DataFramesMeta, Dates, Statistics, Gadfly
import Cairo, Fontconfig

cd(@__DIR__)
include("../utilities.jl")
include("../visualization.jl")

df = CSV.read("../../data/analytical/cows-analytic.csv", DataFrame)

function plot_dist(df::DataFrame, 
                   x::Symbol, 
                   label::String, 
                   fn::String)

  if x in [:ypmlf,:ypmlr,:ypmrf,:ypmrr]
    p = plot(df,
      x = x,
      Geom.histogram,
      Guide.xlabel(label),
      Guide.xticks(ticks = [0,1,2,3,4,5,10,20])
    )
  elseif x == :lactnum
    p = plot(df,
      x = :lactnum,
      Guide.xlabel(label),
      Guide.ylabel("Count"),
      Geom.bar
    )
  elseif x == :mdi
    p = plot(df,
      x = x,
      Geom.histogram,
      Guide.xlabel(label),
      Guide.xticks(ticks = [1,1.4,2,4,6,8,10])
    )
  else
    p = plot(df,
      x = x, 
      Geom.histogram,
      Guide.xlabel(label)
    )
  end
  draw(PNG(fn), p)
end

# -------------------------- Demographics ---------------------------
plot_dist(df, :lactnum, "Lactation Number", "../../resources/figures/dist-lactnum.png")

# ------------------------------ Time -------------------------------
plot_dist(df, :dinmilk, "Days in Milk", "../../resources/figures/dist-dinmilk.png")
plot_dist(df, :interval, "Interval", "../../resources/figures/dist-interval.png")
plot_dist(df, :lastmilkint, "Last Milking Interval", "../../resources/figures/dist-lastmilkint.png")
plot_dist(df, :mdurS, "Milking Duration (Seconds)", "../../resources/figures/dist-mdurS.png")
plot_dist(df, :mdurM, "Milking Duration (Minutes)", "../../resources/figures/dist-mdurM.png")

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

plot_dist(df, 
  :condtot, 
  "Conductivity (Total)", 
  "../../resources/figures/dist-condtot.png"
)

# ------------------------------ Yield ------------------------------
plot_dist(df, 
  :yieldlf, 
  "Yield (Left Front)", 
  "../../resources/figures/dist-yieldlf.png"
)

plot_dist(df, 
  :yieldlr, 
  "Yield (Left Rear)", 
  "../../resources/figures/dist-yieldlr.png"
)

plot_dist(df, 
  :yieldrf, 
  "Yield (Right Front)", 
  "../../resources/figures/dist-yieldrf.png"
)

plot_dist(df, 
  :yieldrr, 
  "Yield (Right Rear)", 
  "../../resources/figures/dist-yieldrr.png"
)

# --- Yield per min ---
plot_dist(df, 
  :ypmlf, 
  "Yield per min (Left Front)", 
  "../../resources/figures/dist-ypmlf.png"
)

plot_dist(df, 
  :ypmlr, 
  "Yield per min (Left Rear)", 
  "../../resources/figures/dist-ypmlr.png"
)

plot_dist(df, 
  :ypmrf, 
  "Yield per min (Right Front)", 
  "../../resources/figures/dist-ypmrf.png"
)

plot_dist(df, 
  :ypmrr, 
  "Yield (Right Rear)", 
  "../../resources/figures/dist-ypmrr.png"
)

# ------------------------------ Flow -------------------------------
plot_dist(df, 
  :flowlf, 
  "Flow (Left Front)", 
  "../../resources/figures/dist-flowlf.png"
)

plot_dist(df, 
  :flowlr, 
  "Flow (Left Rear)", 
  "../../resources/figures/dist-flowlr.png"
)

plot_dist(df, 
  :flowrf, 
  "Flow (Right Front)", 
  "../../resources/figures/dist-flowrf.png"
)

plot_dist(df, 
  :flowrr, 
  "Flow (Right Rear)", 
  "../../resources/figures/dist-flowrr.png"
)

# --- Peak Flow ---
plot_dist(df, 
  :peaklf, 
  "Peak Flow (Left Front)", 
  "../../resources/figures/dist-peaklf.png"
)

plot_dist(df, 
  :peaklr, 
  "Peak Flow (Left Rear)", 
  "../../resources/figures/dist-peaklr.png"
)

plot_dist(df, 
  :peakrf, 
  "Peak Flow (Right Front)", 
  "../../resources/figures/dist-peakrf.png"
)

plot_dist(df, 
  :flowrr, 
  "Peak Flow (Right Rear)", 
  "../../resources/figures/dist-flowrr.png"
)

# ------------------------------ Blood ------------------------------
plot_dist(df, 
  :bloodlf, 
  "Blood Flow (Left Front)", 
  "../../resources/figures/dist-bloodlf.png"
)

plot_dist(df, 
  :bloodlr, 
  "Blood Flow (Left Rear)", 
  "../../resources/figures/dist-bloodlr.png"
)

plot_dist(df, 
  :bloodrf, 
  "Blood Flow (Right Front)", 
  "../../resources/figures/dist-bloodrf.png"
)

plot_dist(df, 
  :bloodrr, 
  "Blood Flow (Right Rear)", 
  "../../resources/figures/dist-bloodrr.png"
)

plot_dist(df,
  :bloodtot,
  "Blood Total",
  "../../resources/figures/dist-bloodtot.png"
)

# ------------------------------ Others -----------------------------
plot_dist(df,
  :mdi,
  "MDi",
  "../../resources/figures/dist-mdi.png"
)