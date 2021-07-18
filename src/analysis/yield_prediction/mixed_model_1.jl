using DataFrames,
      CategoricalArrays,
      GLM,
      MixedModels,
      Plots,
      StatsPlots,
      DataFramesMeta,
      CSV,
      Dates

include("reformatting.jl")

# IMPORT DATA ------------------------------------------------------------------
# move to directory of current file
cd(@__DIR__)
# read file
fname = "../../../data/analytical/cows-analytic.csv"
df = CSV.read(fname, DataFrame)
# add column "weeks in milk"
insertcols!(df, 5, :winmilk => ceil.(Int16, df.dinmilk ./ 7))

# MAKE SUB-DATAFRAME -----------------------------------------------------------
healthy = remove_unhealthydata(df, 7)
healthy[!, :id] = categorical(healthy.id)
healthy[!, :logyield] = log.(healthy.yield)

# PLOTS ------------------------------------------------------------------------
# We will be making an assumption that our error is normally distributed. We
# need to checkif logyield follows a normal distribution.
# ------------------------------------------------------------------------------
scatter(healthy.dinmilk, healthy.yield)
histogram(healthy.logyield)

# SPLIT DATA -------------------------------------------------------------------
#
# Train-test split.
# Train split: All data up until N days prior.
# Test split: All data starting from N days prior
#
# ------------------------------------------------------------------------------
split_date = maximum(healthy.date) - Day(45)
healthy_tr = @subset(healthy, :date .< split_date)
healthy_te = @subset(healthy, :date .>= split_date)

# MIXED-EFFECTS MODEL 1 --------------------------------------------------------
# Initial model: y(n) = a * n^b * exp(-cn)
# Rearranged model: log(y(n)) = log(a) + b*log(n) - c*n
# Variables:
#     Response: :logyield
#     Predictor: :winmilk, :lactnum
# ------------------------------------------------------------------------------
# fit model 
fm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum + (1|id))
model = fit(MixedModel, fm, healthy_tr)
a_, b_, c_ = coef(model)
a, b, c = exp(a_), b_, -c_
# model analysis
n_tr = nrow(healthy_tr)
r² = r2(model)
pred = predict(model)
err = healthy_tr.logyield - pred
mse = sum(err.^2) / (n_tr-2)
scatter(healthy_tr.winmilk, err, c=healthy_tr.lactnum)
# test model
n_te = nrow(healthy_te)
pred = predict(model, healthy_te)
err = healthy_te.logyield - pred
mse = sum(err.^2) / (n_te-2)

# MODEL ANALYTICS --------------------------------------------------------------
#
# It is clear that the model is not a good fit as there are significantly more
# factors causing affecting the yield in cows and better modelling strategy is 
# needed.
#
# Some additional factors include:
#     - lactation number
#     - date
#     - animal [id]
#     - whether the cow has mastitis
# ------------------------------------------------------------------------------
rng = 1:maximum(yield.winmilk)
pred = a * rng.^b .* exp.(-c*rng)
scatter(healthy_tr.winmilk, healthy_tr.yield, label="train")
scatter!(healthy_te.winmilk, healthy_te.yield, label="test")
plot!(rng, pred, lw=3, c=:black, label="model fit")
