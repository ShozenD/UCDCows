using DataFrames,
      GLM,
      Plots,
      StatsPlots,
      DataFramesMeta,
      CSV,
      Dates,
      Random

include("utils.jl")

# IMPORT DATA ------------------------------------------------------------------
# move to directory of current file
cd(@__DIR__)
# read file
fname = "../../../data/analytical/cows-analytic.csv"
df = CSV.read(fname, DataFrame)
# add column "weeks in milk"
insertcols!(df, 5, :winmilk => ceil.(Int16, df.dinmilk ./ 7))

# MAKE SUB-DATAFRAME -----------------------------------------------------------
yield = groupby(df, [:id, :dinmilk])
yield = combine(yield, 
                :lactnum, 
                :yield => sum => :yield,
                :date)
unique!(yield, [:id, :dinmilk, :lactnum, :yield])
yield.logyield = log.(yield.yield .+ 1)
describe(yield)

# PLOTS ------------------------------------------------------------------------
# We will be making an assumption that our error is normally distributed. We
# need to checkif logyield follows a normal distribution.
# ------------------------------------------------------------------------------
histogram(yield.logyield)

# SPLIT DATA BY DATE -----------------------------------------------------------
#
# Train-test split.
# Train split: All data up until 2 weeks prior.
# Test split: All data starting from 2 weeks prior
# ------------------------------------------------------------------------------
split_date = maximum(yield.date) - Day(14)
train_bydate = @subset(yield, :date .< split_date)
test_bydate = @subset(yield, :date .>= split_date)

# RANDOM SPLIT OF DATA ---------------------------------------------------------
#
# Train-test split.
# Train split: Randomly chosen 80% of data
# Test split: Remaining 20% of data not in train split
# ------------------------------------------------------------------------------
sample_size = nrow(yield)
rand_order = randperm(sample_size)
train_rng = rand_order[begin:(floor(Int64, sample_size*0.8))]
train_byrand = yield[train_rng,:]
test_rng = rand_order[(floor(Int64, sample_size*0.8)+1):end]
test_byrand = yield[test_rng,:]


# FIXED-EFFECTS MODEL 1 --------------------------------------------------------
# Initial model: y(n) = a * n^b * exp(-cn)
# Rearranged model: log(y(n)) = log(a) + b*log(n) - c*n
# Variables:
#     Response: :logyield
#     Predictor: :winmilk
# 
# Model fit will be performed on two approaches:
#     1. Data split by date
#     2. Data split randomly
# ------------------------------------------------------------------------------

# Model fit 1 -- data split by date --------------------------------------------
# fit model 
model = lm(@formula(logyield ~ 1 + log(dinmilk) + dinmilk), train_bydate)
print_modelstatistics(model, train_bydate, test_bydate)

# Model fit 2 -- data split randomly -------------------------------------------
# fit model 
model = lm(@formula(logyield ~ 1 + log(dinmilk) + dinmilk), train_byrand)
print_modelstatistics(model, train_byrand, test_byrand)

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
# scatter(yield_tr.winmilk, yield_tr.yield, label="train")
# scatter!(yield_te.winmilk, yield_te.yield, label="test")
# plot!(rng, pred, lw=3, c=:black, label="model fit")
