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

include("utils.jl")

# FUNCTIONS --------------------------------------------------------------------
function remove_unhealthydata(df::DataFrame, Ndays::Integer, verbose::Bool=false)
    # group data frame by animal number / id -----------------------------------
    by_id = groupby(df, :id)
    # remove data where animal is deemed "unhealthy" ---------------------------
    # An animal is considered sick (possibly with mastitis) when there is a gap
    # in the data where the animal's milk yield is not recorded.
    # An animal is considered unhealthy / not fully healthy one week before and 
    # after the animal is sick.
    # --------------------------------------------------------------------------
    healthy = DataFrame()   # healthy data
    for ((id,), group) in pairs(by_id)
        id_group = sort(DataFrame(group), :date)
        dts = unique(id_group.date)
        start_, end_ = minimum(id_group.date), maximum(id_group.date)
        # get sick and unhealthy days ------------------------------------------
        sick_days, unhealthy_days = [], []
        for dt in daterange(start_, end_)
            if issickday(dt, dts)
                push!(sick_days, dt)
            elseif isunhealthyday(dt, sick_days, Ndays)
                push!(unhealthy_days, dt)
            end
        end
        # remove unhealthy days ------------------------------------------------
        healthy_group = @linq id_group |> where(:date .∉ Ref(unhealthy_days))
        # group yield values by day --------------------------------------------
        healthy_group = groupby(healthy_group, :dinmilk) |>
                        grps -> combine(grps, 
                                        :id,
                                        :lactnum, 
                                        :yield => sum => :yield, 
                                        :date) |>
                        comb -> unique(comb, :dinmilk)
        healthy = vcat(healthy, healthy_group)
        if length(sick_days) > 0 && verbose
            println("Animal $id:")
            println("\tSick days: $(length(sick_days))")
            println("\tUnhealthy days: $(length(unhealthy_days))")
            println("\tHealthy days: $(nrow(healthy_group))")
            println(repeat("-", 30))
        end
    end
    return healthy
end

function daterange(start_date, end_date)
    numdays = (end_date-start_date).value
    dts = Vector{Date}(undef, numdays+1)
    for i in 0:numdays
        dts[i+1] = start_date + Day(i)
    end
    return dts
end

issickday(dt, dts) = dt ∉ dts

function isunhealthyday(dt, dts, Ndays)
    for day in dts
        if (dt >= daysminusN(day, Ndays)) & (dt <= daysplusN(day, Ndays))
            return true
        end
    end
    return false
end

daysminusN(dt, N) = dt - Day(N)
daysplusN(dt, N) = dt + Day(N)

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

# MIXED-EFFECTS MODEL 2 --------------------------------------------------------
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
print_modelstatistics(model, train_bydate, test_bydate)

# Model fit 2 -- data split randomly -------------------------------------------
# fit model 
fm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum + (dinmilk|id) + (1|date))
model = fit(MixedModel, fm, train_byrand)
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
# scatter(healthy_tr.winmilk, healthy_tr.yield, label="train")
# scatter!(healthy_te.winmilk, healthy_te.yield, label="test")
# plot!(rng, pred, lw=3, c=:black, label="model fit")
