using DataFrames,
      CategoricalArrays,
      GLM,
      MixedModels,
      Plots,
      StatsPlots,
      DataFramesMeta,
      CSV,
      Dates,
      Random

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
healthy[!, :lactnum] = categorical(healthy.lactnum)
healthy[!, :logyield] = log.(healthy.yield .+ 1)

# PLOTS ------------------------------------------------------------------------
# We will be making an assumption that our error is normally distributed. We
# need to checkif logyield follows a normal distribution.
# ------------------------------------------------------------------------------
scatter(healthy.dinmilk, healthy.yield)
histogram(healthy.logyield)

# SPLIT DATA BY DATE -----------------------------------------------------------
#
# Train-test split.
# Train split: All data up until 2 weeks prior.
# Test split: All data starting from 2 weeks prior
# ------------------------------------------------------------------------------
split_date = maximum(healthy.date) - Day(14)
train_bydate = @subset(healthy, :date .< split_date)
test_bydate = @subset(healthy, :date .>= split_date)

# RANDOM SPLIT OF DATA ---------------------------------------------------------
#
# Train-test split.
# Train split: Randomly chosen 80% of data
# Test split: Remaining 20% of data not in train split
# ------------------------------------------------------------------------------
sample_size = nrow(healthy)
rand_order = randperm(sample_size)
train_rng = rand_order[begin:(floor(Int64, sample_size*0.8))]
train_byrand = healthy[train_rng,:]
test_rng = rand_order[(floor(Int64, sample_size*0.8)+1):end]
test_byrand = healthy[test_rng,:]

# MIXED-EFFECTS MODEL 1 --------------------------------------------------------
# Initial model: y(n) = a * n^b * exp(-cn)
# Rearranged model: log(y(n)) = log(a) + b*log(n) - c*n
# Variables:
#     Response: :logyield
#     Predictor: :winmilk, :lactnum
#
# Model fit will be performed on two approaches:
#     1. Data split by date
#     2. Data split randomly
# ------------------------------------------------------------------------------

# Model fit 1 -- data split by date --------------------------------------------
# fit model 
fm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum + (dinmilk|id) + (1|date))
model = fit(MixedModel, fm, train_bydate)
# model analysis
n_tr = nrow(train_bydate)
pred = predict(model)
err = train_bydate.logyield - pred
mse = sum(err.^2) / (n_tr-dof(model))
# test model
n_te = nrow(test_bydate)
pred = predict(model, test_bydate; new_re_levels=:population)
err = test_bydate.logyield - pred
mse = sum(err.^2) / (n_te-dof(model))

# Model fit 2 -- data split randomly -------------------------------------------
# fit model 
fm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum + (dinmilk|id) + (1|date))
model = fit(MixedModel, fm, train_byrand)
# model analysis
n_tr = nrow(train_byrand)
pred = predict(model)
err = train_byrand.logyield - pred
mse = sum(err.^2) / (n_tr-dof(model))
# test model
n_te = nrow(test_byrand)
pred = predict(model, test_byrand; new_re_levels=:population)
err = test_byrand.logyield - pred
mse = sum(err.^2) / (n_te-dof(model))


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
# rng = 1:maximum(yield.winmilk)
# pred = a * rng.^b .* exp.(-c*rng)
# scatter(healthy_tr.winmilk, healthy_tr.yield, label="train")
# scatter!(healthy_te.winmilk, healthy_te.yield, label="test")
# plot!(rng, pred, lw=3, c=:black, label="model fit")
