using Gadfly: DataFrame, transform, ismissing
using CSV, DataFrames, Gadfly, Statistics, DataFramesMeta, Impute, HypothesisTests

include("../visualization.jl")
include("../utilities.jl")

df = CSV.read("data/analytical/cows-analytic.csv", DataFrame)

lactnum_summary = @by(df, :lactnum, obs = length(:id)) |> x -> sort(x, :lactnum)
df2 = @where(df, :lactnum .<= 2)

# Find cows with abnormal MDi values 
df_abnormal_mdi = @where(df2, :mdi .> 1.4)
id_abnormal_mdi = unique(df_abnormal_mdi.id)

# Filter data for cows with abnormal MDi
df2.abnormal_mdi_flag = [id in id_abnormal_mdi for id in df2.id]
df_abnormal_cows = @where(df2, :abnormal_mdi_flag .== 1)
abnorm_cow_id = unique(df_abnormal_cows.id)

# Plot conductivity trends
set_default_plot_size(12cm, 16cm)
p_cond =  plot_conductivity(df_abnormal_cows, abnorm_cow_id[13], 1.4, 5)
p_yield = plot_yield(df_abnormal_cows, abnorm_cow_id[13], 1.4, 5)
vstack(p_cond, p_yield)

cow = @where(df_abnormal_cows, :id .== abnorm_cow_id[1])

cow.mdi = cow.mdi |> Impute.interp |> Impute.locf

find_headtail(cow.mdi .> 1.3)

### Yeild ###
plot_yield(df_abnormal_cows, abnorm_cow_id[10])

### MDi and Conductivity
#: A boxplot between MDi and Conductivity 
function mctyield(D::DataFrame)
  # Filter out missing values
  D = @where(D, :condrr .!= !ismissing(:condrr))
  D = @where(D, :condrf .!= !ismissing(:condrf))
  D = @where(D, :condlr .!= !ismissing(:condlr))
  D = @where(D, :condlf .!= !ismissing(:condlf))

  Nr = nrow(D)

  D.mct = [minmax_teat(D, i; min=false) for i in 1:Nr]

  y = zeros(Nr)
  z = zeros(Nr)
  negteats = Dict([
    :rr => [2,3,4],
    :rf => [1,3,4],
    :lr => [1,2,4],
    :lf => [1,2,3],
    :NA => [1,2,3,4]
  ])
  posteats = Dict([:rr => 1, :rf => 2, :lr => 3, :lf => 4])
  for i in 1:Nr
    yield = [D.yieldrr[i], D.yieldrf[i], D.yieldlr[i], D.yieldlf[i]]
    y[i] += yield[posteats[D.mct[i]]] 
    z[i] += mean(yield[negteats[D.mct[i]]])
  end
  
  D.mctyield = y
  D.nonmctyield = z

  return D
end

df3 = mctyield(df2)
df2 = @where(df2, :mdi .!= ismissing(:mdi))

df3.nonmctyield
df3.mctyield

plot(df2, x = :mdi, y =:mctyield, Geom.boxplot)
plot(df2, x = :mdi, y =:nonmctyield, Geom.boxplot)

df4 = DataFrame([
  :type => [repeat(["max cond teat"], nrow(df3)); repeat(["non max cond teats"], nrow(df3))],
  :yield => [df3.mctyield; df3.nonmctyield],
  :mdi => repeat(df3.mdi, 2)
])

plot(df4, 
  x = :mdi, y =:yield, 
  color =:type, 
  Geom.boxplot
)

using GLM

m1 = lm(@formula(mdi ~ condmax), df2)

# Get Residual Sum of Squares Σ(yᵢ - f(xᵢ))²
rss = deviance(m1)
mse = rss / nrow(df2)

m2 = lm(@formula(mdi ~ condmax + lastmilkint), df2)
rss = deviance(m2)
mse = rss / nrow(df2)

m3 = lm(@formula(mdi ~ condmax + lastmilkint + bloodtot), df2)
rss = deviance(m3)
mse = rss / nrow(df2)

m4 = lm(@formula(mdi ~ condmax * lastmilkint * bloodtot), df2)
rss = deviance(m4)
mse = rss / nrow(df2)

using MLJ, Statistics, LinearAlgebra

train, test = partition(eachindex(df2.mdi), 0.7, shuffle = true)

df_train, df_test = df2[train,:], df2[test,:]

y_train, X_train = df_train.mdi, df_train[:,[:condmax, :lastmilkint, :bloodtot]]
y_test, X_test = df_test.mdi, df_test[:,[:condmax, :lastmilkint, :bloodtot]]

ols = lm(@formula(mdi ~ condmax * lastmilkint * bloodtot), df2)
ŷ = GLM.predict(ols, X_test)
ŷ = round.(ŷ; digits = 1)

mean(skipmissing((y_test - ŷ).^2))
mean(skipmissing(abs.(y_test - ŷ)))

dfres = DataFrame([
  :y => y_test,
  :yhat => ŷ
])

set_default_plot_size(16cm, 10cm)

df3 = @where(df2, :mdi .!= ismissing(:mdi))
plot(df3,
  x = :mdi, y = :condmax,
  Geom.boxplot,
  Guide.xticks(ticks=[1,1.4,2,3,4,5,10])
)

plot(df3,
  x = :mdi, y = :yieldmin,
  Geom.boxplot,
  Guide.xticks(ticks=[1,1.4,2,3,4,5,10])
)

plot(dfres,
  x = :y, y = :yhat,
  Geom.point,
  size = [0.5mm],
  Guide.xticks(ticks=[1,1.4,2,3,4,5,10])
)