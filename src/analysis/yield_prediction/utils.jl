using MixedModels,
      GLM,
      Statistics,
      DataFrames,
      DataFramesMeta

# DATA SPLITTER ----------------------------------------------------------------

# MODEL ANALYZER ---------------------------------------------------------------
function print_modelstatistics(model::LinearMixedModel, 
                               train_data::DataFrame,
                               test_data::DataFrame)
    
    # model statistics
    println("Model Statistics:")
    display(model)

    # train data
    degree_of_freedom = dof(model)
    n_observation = nobs(model)
    pred = predict(model, train_data)
    error = train_data.logyield - pred
    train_sse = sum(error .^ 2)
    train_mse = train_sse / dof_residual(model)
    train_mpwrss = pwrss(model) / dof_residual(model)
    println("\nTrain data:")
    println("Degree of freedom: $degree_of_freedom")
    println("Number of observations: $n_observation")
    println("Mean squared error: $(round(train_mse, digits=4))")
    println("Mean squared penalized, weighted residual: $(round(train_mpwrss, digits=4))")

    # test data
    n_observation = nrow(test_data)
    pred = predict(model, test_data)
    error = test_data.logyield - pred
    test_sse = sum(error .^ 2)
    test_mse = test_sse / (n_observation - degree_of_freedom)
    println("\nTest data statistics:")
    println("Number of observations: $n_observation")
    println("Mean Squared Error: $(round(test_mse, digits=4))")
end

function print_modelstatistics(model::StatsModels.TableRegressionModel,
                               train_data::DataFrame,
                               test_data::DataFrame)
    
    # model statistics
    println("Model Statistics:")
    display(model)

    # train data
    degree_of_freedom = dof(model)
    n_observation = nobs(model)
    r² = r2(model)
    pred = predict(model, train_data)
    error = train_data.logyield - pred
    train_sse = sum(error .^ 2)
    train_mse = train_sse / dof_residual(model)
    println("\nTrain data:")
    println("Degree of freedom: $degree_of_freedom")
    println("Number of observations: $n_observation")
    println("R²: $r²")
    println("Mean squared error: $(round(train_mse, digits=4))")

    # test data
    n_observation = nrow(test_data)
    pred = predict(model, test_data)
    error = test_data.logyield - pred
    test_sse = sum(error .^ 2)
    test_mse = test_sse / (n_observation - degree_of_freedom)
    println("\nTest data statistics:")
    println("Number of observations: $n_observation")
    println("Mean Squared Error: $(round(test_mse, digits=4))")
end