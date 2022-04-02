# function gridsearch(df_healthy::DataFrame,
#                     df_sick::DataFrame,
#                     list::Union{Vector{T}, UnitRange{T}},
#                     criterion::Symbol = :mdi,
#                     mdi_threshold::AbstractFloat = 1.4;
#                     fm::FormulaTerm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum + status + group + normalized_GPTAM),
#                     split_by::Symbol = :date,
#                     split_date::Union{Date, Nothing} = nothing, 
#                     train_size::Union{Real, Nothing} = nothing, 
#                     test_size::Union{Real, Nothing} = nothing, 
#                     random_state::Union{Integer, Nothing} = nothing) where T<:Integer
#     # Grid search through list
#     len = length(list)
#     mae_list = Dict("train" => Vector{Float64}(undef, len),
#                     "test" => Vector{Float64}(undef, len))
#     mpe_list = Dict("train" => Vector{Float64}(undef, len),
#                     "test" => Vector{Float64}(undef, len))
#     coef_list = Dict("status (unhealthy)" => Vector{Float64}(undef, len),
#                      "group (sick)" => Vector{Float64}(undef, len))
#     for (i, n) in enumerate(list)
#         mae_train, mae_test, mpe_train, mpe_test, coefs, _ = categorize_and_fit(
#             df_healthy, df_sick, n, criterion, mdi_threshold, 
#             fm=fm, split_by=split_by, split_date=split_date, train_size=train_size, 
#             test_size=test_size, random_state=random_state
#         )
#         mae_list["train"][i] = mae_train
#         mpe_list["train"][i] = mpe_train
#         mae_list["test"][i] = mae_test
#         mpe_list["test"][i] = mpe_test
#         coef_list["status (unhealthy)"][i] = coefs["status (unhealthy)"]
#         coef_list["group (sick)"][i] = coefs["group (sick)"]
#     end
#     # Return the best parameter along with its results
#     idx = argmin(mpe_list["test"])
#     best_n = list[idx]
#     best_mae = (train = mae_list["train"][idx], test = mae_list["test"][idx])
#     best_mpe = (train = mpe_list["train"][idx], test = mpe_list["test"][idx])
#     best_coef = Dict("status (unhealthy)" => coef_list["status (unhealthy)"][idx],
#                      "group (sick)" => coef_list["group (sick)"][idx])
#     return (n = best_n, mae = best_mae, mpe = best_mpe, coef = best_coef), 
#            (mae = mae_list, mpe = mpe_list, coef = coef_list)
# end
# TODO: Fix case where subdf_train or subdf_test has zero rows
function categorize_and_fit(df_healthy::DataFrame, 
                            df_sick::DataFrame,
                            n::Integer,
                            criterion::Symbol,
                            mdi_threshold::AbstractFloat;
                            split_by::Symbol = :proportion,
                            split_date::Union{Date, Nothing} = nothing, 
                            train_size::Union{Real, Nothing} = 0.8, 
                            test_size::Union{Real, Nothing} = 0.2, 
                            random_state::Union{Integer, Nothing} = nothing)
    # --- Categorize data into groups and health status ---
    df = categorize_data(df_healthy, df_sick, before=2, after=n, criterion=criterion, mdi_threshold=mdi_threshold)
    df[ismissing.(df.logyield), :logyield] .= 0.0       # Fill missing logyield values with 0
    df[df.status .== "sick", :status] .= "unhealthy"    # Sick and unhealthy status considered as unhealthy

    # ----- Model Fitting -----
    # Split model to train-test sets
    df_train, df_test = train_test_split(df, split_by, split_date=split_date, train_size=train_size,
                                         test_size=test_size, random_state=random_state)
    cow_ids = unique(df_train.id)
    results = []
    for cow_id in cow_ids
        subdf_train = @subset(df_train, :id .== cow_id)
        subdf_test = @subset(df_test, :id .== cow_id)
        nrow(subdf_train) > 0 || continue
        nrow(subdf_test) > 0 || continue

        group = unique(subdf_train.group)[1]
        if group == "sick"
            "healthy" ∈ subdf_train.status || continue
            "unhealthy" ∈ subdf_train.status || continue
            "healthy" ∈ subdf_test.status || continue
            "unhealthy" ∈ subdf_test.status || continue
        end
        fm = group == "sick" ? @formula(logyield ~ 1 + log(dinmilk) + dinmilk + status) : @formula(logyield ~ 1 + log(dinmilk) + dinmilk)
        result = model_fit(subdf_train, subdf_test, fm)
        push!(results, result)
    end

    return vcat(results...)
end

function model_fit(df_train::AbstractDataFrame, df_test::AbstractDataFrame, fm::FormulaTerm)
    group = unique(df_train.group)[1]
    @assert (length ∘ unique)(df_train.id) == 1
    @assert (length ∘ unique)(df_train.group) == 1
    @assert (length ∘ unique)(df_test.id) == 1
    @assert (length ∘ unique)(df_test.group) == 1
    @assert unique(df_train.id)[1] == unique(df_test.id)[1]
    @assert unique(df_train.group)[1] == unique(df_test.group)[1]
    @assert (group == "healthy" && Term(:status) ∉ fm.rhs) || (group == "sick" && Term(:status) ∈ fm.rhs)

    cow_id = unique(df_train.id)[1]
    train_size = nrow(df_train)
    test_size = nrow(df_test)
    if train_size > 0
        model = fit(LinearModel, fm, df_train)
        β₀, β₁, β₂ = coef(model)[1:3]
        β₃ = (length ∘ coef)(model) == 4 ? coef(model)[4] : missing
        yield_train_pred = predict(model, df_train) |> x -> exp.(x) |> x -> convert.(Float64, x)
        mae_train = meanad(yield_train_pred, df_train.yield)
        mpe_train = compute_mpe(yield_train_pred, df_train.yield, transform = :none)
    else
        β₀ = missing
        β₁ = missing
        β₂ = missing
        β₃ = missing
        mae_train = missing
        mpe_train = missing
    end

    if train_size > 0 && test_size > 0
        yield_test_pred = predict(model, df_test) |> x -> exp.(x) |> x -> convert.(Float64, x)
        mae_test = meanad(yield_test_pred, df_test.yield)
        mpe_test = compute_mpe(yield_test_pred, df_test.yield, transform = :none)
    else
        mae_test = missing
        mpe_test = missing
    end

    return DataFrame(:id => cow_id,
                     :group => group,
                     :trainSize => train_size,
                     :testSize => test_size,
                     :β₀ => β₀,
                     :β₁ => β₁,
                     :β₂ => β₂,
                     :β₃ => β₃,
                     :MAETrain => mae_train,
                     :MAETest => mae_test,
                     :MPETrain => mpe_train,
                     :MPETest => mpe_test)
end

"""
    train_test_split(df, by[; split_date, train_size, test_size, random_state])

Split data into train and test sets.

# Arguments
- `df::DataFrame`
- `by::Symbol`

# Keyword Arguments
- `split_date::Union{Date, Nothing}`
- `train_size::Union{Real, Nothing}`
- `test_size::Union{Real, Nothing}`
- `random_state::Union{Integer, Nothing}`
"""
function train_test_split(df::DataFrame, by::Symbol; 
                          split_date::Union{Date, Nothing} = nothing, 
                          train_size::Union{Real, Nothing} = nothing, 
                          test_size::Union{Real, Nothing} = nothing, 
                          random_state::Union{Integer, Nothing} = nothing)
    @assert by ∈ [:date, :random, :proportion]

    if by == :date
        @assert !isnothing(split_date)
        df_train = @subset(df, :date .< split_date)
        df_test = @subset(df, :date .≥ split_date)
    elseif by == :proportion
        @assert !isnothing(train_size)
        @assert !isnothing(test_size)
        @assert train_size > 0
        @assert test_size > 0
        train_prop = train_size / (train_size + test_size)
        subdf_by_cowid = groupby(df, [:id, :lactnum])
        df_train_list = Vector{DataFrame}()
        df_test_list = Vector{DataFrame}()
        for subdf in subdf_by_cowid
            subdf = DataFrame(subdf)
            row_count = nrow(subdf)
            T = typeof(row_count)
            subdf_train_size = ceil(T, row_count * train_prop)
            subdf_train = subdf[1:subdf_train_size, :]
            subdf_test = subdf[(subdf_train_size+1):end, :]
            push!(df_train_list, subdf_train)
            push!(df_test_list, subdf_test)
        end
        df_train = vcat(df_train_list...)
        df_test = vcat(df_test_list...)
    else
        @assert !isnothing(train_size)
        @assert !isnothing(test_size)
        @assert train_size > 0
        @assert test_size > 0
        n = nrow(df)
        train_size = floor(typeof(n), train_size * n / (train_size+test_size))
        isnothing(random_state) || Random.seed!(random_state)
        nₚ = randperm(n)
        rng_tr = nₚ[1:(train_size)]
        rng_te = nₚ[(train_size+1):end]
        df_train = df[rng_tr,:]
        df_test = df[rng_te,:]
    end
    return df_train, df_test
end

@doc raw"""
    compute_mpe(ŷ, y[; transform])

Computes Mean Proportion Error (MPE), defined as:

``MPE =  \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y_i}| / y_i``

# Arguments
- `ŷ::AbstractVector{T} where T<:Number`: Predicted values.
- `y::AbstractVector{T} where T<:Number`: True values.

# Keyword Arguments
- `transform::Symbol`: (Default: `:none`) Transform type of ŷ and y. Accepted values are
  `:none` and `:log`.

# Returns
- `mpe::T`: Mean Proportion Error.
"""
function compute_mpe(ŷ::AbstractVector, y::AbstractVector; transform::Symbol = :none) where T<:Number
    @assert transform ∈ [:none, :log]
    @assert length(ŷ) == length(y)
    if transform == :log
        ŷ = exp.(ŷ)
        y = exp.(y)
    end
    return mean(abs.(y - ŷ) ./ y)
end