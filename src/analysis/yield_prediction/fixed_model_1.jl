using DataFrames,
      GLM,
      Plots,
      StatsPlots,
      DataFramesMeta,
      CSV,
      Dates

# IMPORT DATA ------------------------------------------------------------------
# move to directory of current file
cd(@__DIR__)
# read file
fname = "../../../data/analytical/cows-analytic.csv"
df = CSV.read(fname, DataFrame)
# add column "weeks in milk"
insertcols!(df, 5, :winmilk => ceil.(Int16, df.dinmilk ./ 7))

# MAKE SUB-DATAFRAME -----------------------------------------------------------
yield = groupby(df, [:id, :winmilk])
yield = combine(yield, 
                :lactnum, 
                :yield => sum => :yield,
                :date)
unique!(yield, [:id, :winmilk, :lactnum, :yield])
yield.logyield = log.(yield.yield)
describe(yield)

# PLOTS ------------------------------------------------------------------------
# We will be making an assumption that our error is normally distributed. We
# need to checkif logyield follows a normal distribution.
# ------------------------------------------------------------------------------
histogram(yield.logyield)

# SPLIT DATA -------------------------------------------------------------------
#
# Train-test split.
# Train split: All data up until 2 weeks prior.
# Test split: All data starting from 2 weeks prior
#
# Note: Data from the date 5/24/2021 cannot be used in train data as it causes
# multicollinearity issues.
# ------------------------------------------------------------------------------
split_date = maximum(yield.date) - Day(14)
xdate = Date(2021, 5, 24)
yield_tr = @subset(yield, :date .< split_date, :date .!= xdate)
yield_te = vcat(@subset(yield, :date .>= split_date),
                @subset(yield, :date .== xdate))

# FIXED-EFFECTS MODEL 1 --------------------------------------------------------
# Initial model: y(n) = a * n^b * exp(-cn)
# Rearranged model: log(y(n)) = log(a) + b*log(n) - c*n
# Variables:
#     Response: :logyield
#     Predictor: :winmilk
# ------------------------------------------------------------------------------
# fit model 
model = lm(@formula(logyield ~ 1 + log(winmilk) + winmilk), yield_tr)
a_, b_, c_ = coef(model)
a, b, c = exp(a_), b_, -c_
# model analysis
n_tr = nrow(yield_tr)
r² = r2(model)
pred = predict(model)
err = yield_tr.logyield - pred
mse = sum(err.^2) / (n_tr-2)
scatter(yield_tr.winmilk, err, c=yield_tr.lactnum)
# test model
n_te = nrow(yield_te)
pred = predict(model, yield_te)
err = yield_te.logyield - pred
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
scatter(yield_tr.winmilk, yield_tr.yield, label="train")
scatter!(yield_te.winmilk, yield_te.yield, label="test")
plot!(rng, pred, lw=3, c=:black, label="model fit")
