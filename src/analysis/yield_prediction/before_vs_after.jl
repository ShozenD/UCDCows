# Analyze the difference in curves before and after mastitis infection in cows.
# ------------------------------------------------------------------------------
#
# Steps:
#   1. For each cow, split the data into two data frames: one containing data
#      before mastitis, and the other one after the infection.
#      (Optional): Remove cows that have always been healthy, ie. never 
#                  had any cases of mastitis. Only do this if the amount of 
#                  data after the removal is still reasonable.
#   2. Build 2 fixed-effect models based on the 2 different sets of data from 
#      (1).
#   3. For each cow, make predictions on the entire time series using both 
#      models. Then, determine whether one curve is lower than the other.
#   4. Model inferences. Metrics to check:
#           a. MSE: MSE from after-mastitis model should be lower when
#              predicting after-mastitis yield. The same concept applies to
#              before-mastitis model.
#           b. MAE: While MSE is a good metric to use, if the errors have
#              magnitudes less than 1, it is not a good indicator.
#           c. Average differences between before vs after models. This will
#              be used to determine whether the average yield before mastitis
#              is actually higher than after mastitis.
# ------------------------------------------------------------------------------

# Import packages --------------------------------------------------------------
using DataFrames, 
      DataFramesMeta,
      Dates,
      CSV,
      CategoricalArrays,
      GLM,
      Plots, 
      Statistics

include("utils.jl")

# Import data ------------------------------------------------------------------
# Move to directory of current file
cd(@__DIR__)
# Read file
fname = "../../../data/analytical/cows-analytic.csv"
df = CSV.read(fname, DataFrame)
# Format column types
df[!, :id] = categorical(df.id)
df[!, :lactnum] = categorical(df.lactnum)

# Parameter Search -------------------------------------------------------------
# Find the `criterion` and `n_days` with the largest expected difference
# between healthy and sick cows
split_date = maximum(df.date) - Day(21)
train_df = @subset(df, :date .< split_date)
days_range = collect(1:14)
expected_diff = gridsearch_expected_difference(train_df, days_range)
plot_differences(expected_diff, days_range)

# Split data into healthy and sick data frames ---------------------------------
healthy_data, sick_data, by_cows = splitbyperiod(df, 5, :sick)

# Split data by date -----------------------------------------------------------
#
# Train-test split
# Train split: All data up until 3 weeks prior.
# Test split: All data starting from 3 weeks prior.
train_healthy_bydate = @subset(healthy_data, :date .< split_date)
train_sick_bydate = @subset(sick_data, :date .< split_date)
test_healthy_bydate = @subset(healthy_data, :date .>= split_date)
test_sick_bydate = @subset(sick_data, :date .>= split_date)

# Fixed-Effects Model ----------------------------------------------------------
# Equation formula
fm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum)
# Fit healthy training data
model_healthy = fit(LinearModel, fm, train_healthy_bydate)
model_sick = fit(LinearModel, fm, train_sick_bydate)
# Print model statistics
print_modelstatistics(model_healthy, train_healthy_bydate, test_healthy_bydate)
print_modelstatistics(model_sick, train_sick_bydate, test_sick_bydate)

# Plot the curves for healthy model vs sick model ------------------------------
df_seq = makedataset(by_cows[1].healthy_data)
pred_healthy, pred_sick = make_prediction(model_healthy, model_sick, df_seq)
plot([pred_healthy pred_sick], label=["healthy" "sick"], 
     ylim=[0, 150], ylabel="Milk Yield", xlabel="Days In Milk")