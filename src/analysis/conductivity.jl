using CSV: length
using Base: Float64
using Gadfly: DataFrame, transform, line
using CSV, DataFrames, Gadfly, Statistics, DataFramesMeta, Impute, HypothesisTests

include("../visualization.jl")
include("../utilities.jl")

df = CSV.read("data/analytical/cows-analytic.csv", DataFrame)

thmdi = 1.4

df2 = @where(df, :lactnum .<= 2)
abnrmcow = @where(df, :mdi .> 1.4)
abnrmcowid = unique(abnrmcow.id)

Yield = Array{Float64,2}(undef, length(abnrmcowid), 4)
MCTeat = Vector{Symbol}(undef, length(abnrmcowid))
abnrmcowid
for i in 1:length(abnrmcowid)
  cow = @where(abnrmcow, :id .== abnrmcowid[i])
  cow.mdi = linear_interpolation(cow.mdi)

  h, t = find_headtail(cow.mdi .> thmdi)
  h₁, t₁ = h[1], t[1]

  try
    maxcondteat = minmax_teat(cow, h₁; min = false)

    yieldrr = mean(cow.yieldrr[h₁:t₁])
    yieldrf = mean(cow.yieldrf[h₁:t₁])
    yieldlr = mean(cow.yieldlr[h₁:t₁])
    yieldlf = mean(cow.yieldlf[h₁:t₁])

    Yield[i,:] = [yieldrr, yieldrf, yieldlr, yieldlf]
    MCTeat[i] = maxcondteat
  catch
    Yield[i,:] = [-999,-999,-999,-999]
    MCTeat[i] = :NA
  end
end

Yield
AvgLowCondYield = Vector{Float64}(undef, length(abnrmcowid))
HighCondYield = Vector{Float64}(undef, length(abnrmcowid))

for i in 1:length(abnrmcowid)
  negteats = Dict([
    :rr => [2,3,4],
    :rf => [1,3,4],
    :lr => [1,2,4],
    :lf => [1,2,3],
    :NA => [1,2,3,4]
  ])
  posteats = Dict([:rr => 1, :rf => 2, :lr => 3, :lf => 4, :NA => 4])

  AvgLowCondYield[i] = mean(Yield[i,negteats[MCTeat[i]]])
  HighCondYield[i] = Yield[i,posteats[MCTeat[i]]]
end

AvgLowCondYield
HighCondYield

dfy = DataFrame([
  :AvgLowCondYield => AvgLowCondYield,
  :HighCondYield => HighCondYield
])

dfy = @where(dfy, :AvgLowCondYield .!= -999)

plot(dfy,
  layer(x = :HighCondYield, Geom.density, color=[colorant"#de425b"]),
  layer(x = :AvgLowCondYield, Geom.density, color=[colorant"#488f31"]),
  Guide.xlabel("Yield"),
  Guide.ylabel("Density"),
  Theme(background_color="white")
)

mean(dfy.AvgLowCondYield)
mean(dfy.HighCondYield)

# Parametric
UnequalVarianceTTest(dfy.AvgLowCondYield, dfy.HighCondYield)

# Non-parametric
MannWhitneyUTest(dfy.AvgLowCondYield, dfy.HighCondYield)