# ========== Analysis with cows where first N days are recorded ============================
# Diff from v1:
#   - Uses dummy variables and have everything fitted into one linear regression equation
#     instead of having multiple equations
#   

# ===== Import Packages =====
using DataFrames,
      DataFramesMeta,
      CSV,
      CategoricalArrays,
      Dates,
      Statistics,
      Plots,
      GLM,
      LaTeXStrings,
      Random,
      StatsBase,
      AverageShiftedHistograms,
      Distributions

# ========== Functions ==========
include("data_cleaning.jl")
include("model_fit.jl")
include("visualization.jl")

## ========== Workflow ==========
# ----- Import Data -----
@info "Running: Data Cleaning"
# Move to directory of current file
cd(@__DIR__)
# Read milk production data file
path = "../../../data/analytical/cows-analytic.csv"
df = CSV.read(path, DataFrame)
df[!, :id] = categorical(df.id)
df[!, :lactnum] = categorical(df.lactnum)
# Read genomic info data file 
# 190 out of 2401 cows have GPTAM values before data cleaning
# 55 out of 695 cows have GPTAM values after data cleaning
path = "../../../data/misc/genomic_info.csv"
genomic_info = CSV.read(path, DataFrame) |> x -> x[!, [:id, :GPTAM]]
genomic_info[!, :GPTAM] = convert(Vector{Float64}, genomic_info[!, :GPTAM])
mean_GPTAM = mean(genomic_info.GPTAM)
std_GPTAM = std(genomic_info.GPTAM)
# Left join milk_summary info with GPTAM info
milk_summary = groupby(df, [:id, :date])|>      # Get daily total milk
               grps -> combine(grps, :lactnum, :yield => sum => :yield) |>
               unique |>
               milk -> groupby(milk, :id) |>    # Get average daily milk
               grps -> combine(grps, :lactnum, :yield => mean => :mean_yield) |>
               unique |>
               df -> leftjoin(df, genomic_info, on = :id) |>    # Left join with GPTAM info
               df -> @subset(df, :lactnum .∈ Ref([1,2,3]))
# Impute GPTAM values using linear regression
fm₀ = @formula(GPTAM ~ 1 + lactnum + mean_yield)
model = fit(LinearModel, fm₀, milk_summary)
idx = ismissing.(milk_summary.GPTAM)
milk_summary[idx, :GPTAM] = predict(model, milk_summary[idx,:])
milk_summary[!, :GPTAM] = convert.(Float64, milk_summary[!, :GPTAM])
milk_summary[!, :normalized_GPTAM] = (milk_summary.GPTAM .- mean_GPTAM) / std_GPTAM
# Left join milk summary with original dataset
df = leftjoin(df, milk_summary, on = [:id, :lactnum])

## ----- Data Processing -----
# Keep data of cows with records of their first 30 days in milk, discard the rest of the
# data.
filtered, summary_data = filter_cows(df)
# -- Remove certain cows that have missing teats --
# Cows: 9064 (LR), 49236 (RF), 7984 (RF), 42130 (LR), and 48695 (LR)
filtered = @subset(filtered, :id .∉ Ref([9064, 49236, 7984, 42130, 48695]))
# Split data into healthy and sick
mdi_threshold₁ = 1.4
mdi_threshold₂ = 1.8
df_healthy₁, df_sick₁ = splitbyhealth(filtered, 30, criterion=:mdi, threshold=mdi_threshold₁)
df_healthy₂, df_sick₂ = splitbyhealth(filtered, 30, criterion=:mdi, threshold=mdi_threshold₂)
# -- Remove "healthy" cows with MDi above threshold at any point of milking -- 
# Only keep fully healthy cows
cows = @subset(df_healthy₁, :mdi .≥ mdi_threshold₁) |> x -> unique(x.id)
df_healthy₁ = @subset(df_healthy₁, :id .∉ Ref(cows))
cows = @subset(df_healthy₂, :mdi .≥ mdi_threshold₂) |> x -> unique(x.id)
df_healthy₂ = @subset(df_healthy₂, :id .∉ Ref(cows))
# Remove "healthy" cows with gaps
df_healthy₁ = removecowswithgaps(df_healthy₁)
df_healthy₂ = removecowswithgaps(df_healthy₂)
# --- Aggregate data ---
df_healthy₁ = aggregate_data(df_healthy₁)       # Data where all cows have MDi<1.4 at all times
df_healthy₂ = aggregate_data(df_healthy₂)       # Data where all cows have MDi<1.8 at all times
df_sick₁ = aggregate_data(df_sick₁)             # Data where cows have MDi≥1.4 at certain times
df_sick₂ = aggregate_data(df_sick₂)             # Data where cows have MDi≥1.8 at certain times
# --- Split to train and test sets ---
train_healthy₁, test_healthy₁ = train_test_split(df_healthy₁, :date, split_date = Date(2021,11,1))
train_healthy₂, test_healthy₂ = train_test_split(df_healthy₂, :date, split_date = Date(2021,11,1))
train_sick₁, test_sick₁ = train_test_split(df_sick₁, :date, split_date = Date(2021,11,1))
train_sick₂, test_sick₂ = train_test_split(df_sick₂, :date, split_date = Date(2021,11,1))

## ----- Grid search for best model fit -----
@info "Running: Grid search"
dinmilk_range = 0:100
split_date = Date(2021, 10, 1)
# Grid search for mdi_threshold = 1.4
best₁, list₁ = gridsearch(train_healthy₁, train_sick₁, dinmilk_range, :mdi, mdi_threshold₁,
                          split_by=:date, split_date = split_date)
p1 = plot_error(dinmilk_range, list₁, :mpe)
p2 = plot(dinmilk_range, abs.(list₁.coef["status (unhealthy)"]), label = "β₄ when vs mid-high MDi cows",
          xlabel = "k days after event", ylabel = "|coef.|", 
          title = "Coef. magnitude for status=unhealthy (β₄)", 
          titlefontsize = 7, xguidefontsize = 8, yguidefontsize = 8, legend=:topright)
p3 = plot(dinmilk_range, abs.(list₁.coef["group (sick)"]), label = "β₅ when vs mid-high MDi cows",
          xlabel = "k days after event", ylabel = "|coef.|",
          title = "Coef. magnitude for group=sick (β₅)",
          titlefontsize = 7, xguidefontsize = 8, yguidefontsize = 8, legend=:topleft)
# Grid search for mdi_threshold = 1.8
best₂, list₂ = gridsearch(train_healthy₁, train_sick₂, dinmilk_range, :mdi, mdi_threshold₂,
                          split_by=:date, split_date = split_date)
plot_error!(p1, dinmilk_range, list₂, :mpe, :black)
vline!(p1, [best₁.n, best₂.n], color = :red, label = "")
plot!(p2, dinmilk_range, abs.(list₂.coef["status (unhealthy)"]), label = "β₄ when vs high MDi cows")
plot!(p3, dinmilk_range, abs.(list₂.coef["group (sick)"]), label = "β₅ when vs high MDi cows")
# Overall plot
p4 = plot(p2, p3, layout=(2,1))
p5 = plot(p1, p4, layout=(1,2))
savefig(p5, "gridsearch.png")

## Plot an example of true vs predicted milk yield for a specific cow in each category.
# Healthy: 8244, 8906, 8329, 8169, 7796
# Mid high: 44923, 45890, 45808, 45440
# High: 45808, 9008, 9007
fm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum + status + group + normalized_GPTAM)
_, _, _, _, _, model = categorize_and_fit(
    df_healthy₁, df_sick₁, 1, :mdi, mdi_threshold₁, fm = fm, split_by = :date, split_date = split_date
)
# Normal MDI
subdf1 = @subset(df_healthy₁, :id .== 7796) |> x -> sort(x, :date)
subdf2 = @subset(df_healthy₁, :id .== 8906) |> x -> sort(x, :date)
subdf3 = @subset(df_healthy₁, :id .== 8329) |> x -> sort(x, :date)
p1 = true_vs_fitted(subdf1, model, split_date = split_date, mdi_threshold = mdi_threshold₁) |> plt -> ylims!(0,200)
p2 = true_vs_fitted(subdf2, model, split_date = split_date, mdi_threshold = mdi_threshold₁) |> plt -> ylims!(0,200)
p3 = true_vs_fitted(subdf3, model, split_date = split_date, mdi_threshold = mdi_threshold₁) |> plt -> ylims!(0,200)
p4 = plot(p1, p2, p3, layout = (1,3))
# Mid High MDI
subdf1 = @subset(df_sick₁, :id .== 45808) |> x -> sort(x, :date)
subdf2 = @subset(df_sick₁, :id .== 9008) |> x -> sort(x, :date)
subdf3 = @subset(df_sick₁, :id .== 9007) |> x -> sort(x, :date)
p1 = true_vs_fitted(subdf1, model, split_date = split_date, mdi_threshold = mdi_threshold₁) |> plt -> ylims!(0,200)
p2 = true_vs_fitted(subdf2, model, split_date = split_date, mdi_threshold = mdi_threshold₁) |> plt -> ylims!(0,200)
p3 = true_vs_fitted(subdf3, model, split_date = split_date, mdi_threshold = mdi_threshold₁) |> plt -> ylims!(0,200)
p5 = plot(p1, p2, p3, layout = (1,3))
# Model fit with MDI threshold 1.8
_, _, _, _, _, model = categorize_and_fit(
    df_healthy₁, df_sick₂, 1, :mdi, mdi_threshold₂, fm = fm, split_by = :date, split_date = split_date
)
# Normal MDI
subdf1 = @subset(df_healthy₁, :id .== 8244) |> x -> sort(x, :date)
subdf2 = @subset(df_healthy₁, :id .== 8906) |> x -> sort(x, :date)
subdf3 = @subset(df_healthy₁, :id .== 8329) |> x -> sort(x, :date)
p1 = true_vs_fitted(subdf1, model, split_date = split_date, mdi_threshold = mdi_threshold₁) |> plt -> ylims!(0,200)
p2 = true_vs_fitted(subdf2, model, split_date = split_date, mdi_threshold = mdi_threshold₁) |> plt -> ylims!(0,200)
p3 = true_vs_fitted(subdf3, model, split_date = split_date, mdi_threshold = mdi_threshold₁) |> plt -> ylims!(0,200)
p6 = plot(p1, p2, p3, layout = (1,3))
# Mid High MDI
subdf1 = @subset(df_sick₂, :id .== 45808) |> x -> sort(x, :date)
subdf2 = @subset(df_sick₂, :id .== 9008) |> x -> sort(x, :date)
subdf3 = @subset(df_sick₂, :id .== 9007) |> x -> sort(x, :date)
p1 = true_vs_fitted(subdf1, model, split_date = split_date, mdi_threshold = mdi_threshold₂) |> plt -> ylims!(0,200)
p2 = true_vs_fitted(subdf2, model, split_date = split_date, mdi_threshold = mdi_threshold₂) |> plt -> ylims!(0,200)
p3 = true_vs_fitted(subdf3, model, split_date = split_date, mdi_threshold = mdi_threshold₂) |> plt -> ylims!(0,200)
p7 = plot(p1, p2, p3, layout = (1,3))

## ----- Search for `k` days for best model fit -----
@info "Running: Repeated Grid Search"
dinmilk_range = 0:100
n_trials = 100
random_states = sample(1:10000, n_trials, replace=false)
n_days₁ = Vector{Int64}(undef, n_trials); n_days₂ = Vector{Int64}(undef, n_trials)
maes₁ = Vector{Float64}(undef, n_trials); maes₂ = Vector{Float64}(undef, n_trials)
mpes₁ = Vector{Float64}(undef, n_trials); mpes₂ = Vector{Float64}(undef, n_trials)
coefs₁ = Dict{String, Vector{Float64}}(); coefs₂ = Dict{String, Vector{Float64}}()
coefs₁["status (unhealthy)"] = Float64[]; coefs₂["status (unhealthy)"] = Float64[]
coefs₁["group (sick)"] = Float64[]; coefs₂["group (sick)"] = Float64[]
for (i, random_state) in enumerate(random_states)
    mod(i,100) == 0 ? (@info "Repeated Grid Search: Loop $i") : nothing
    b₁, l₁ = gridsearch(train_healthy₁, train_sick₁, dinmilk_range, :mdi, mdi_threshold₁,
                              split_by=:random, train_size=0.9, test_size=0.1, random_state=random_state)
    b₂, l₂ = gridsearch(train_healthy₁, train_sick₂, dinmilk_range, :mdi, mdi_threshold₂,
                              split_by=:random, train_size=0.9, test_size=0.1, random_state=random_state)
    n_days₁[i] = b₁.n; n_days₂[i] = b₂.n
    maes₁[i] = b₁.mae.test; maes₂[i] = b₂.mae.test
    mpes₁[i] = b₁.mpe.test; mpes₂[i] = b₂.mpe.test
    push!(coefs₁["status (unhealthy)"], b₁.coef["status (unhealthy)"])
    push!(coefs₂["status (unhealthy)"], b₂.coef["status (unhealthy)"])
    push!(coefs₁["group (sick)"], b₁.coef["group (sick)"])
    push!(coefs₂["group (sick)"], b₂.coef["group (sick)"])
end

d₁ = unique(n_days₁) |> sort!; c₁ = [count(==(i), n_days₁) for i in d₁]
d₂ = unique(n_days₂) |> sort!; c₂ = [count(==(i), n_days₂) for i in d₂]
p1 = plot(title="β₄ when vs mid-high MDi cows", titlefontsize=10, legend=:outerright) 
p2 = plot(title="β₄ when vs high MDi cows", titlefontsize=10, legend=:outerright)
p3 = plot(title="β₅ when vs mid-high MDi cows", titlefontsize=10, legend=:outerright) 
p4 = plot(title="β₅ when vs high MDi cows", titlefontsize=10, legend=:outerright)
p5 = plot(title="MAE when vs mid-high MDi cows", titlefontsize=10, legend=:outerright) 
p6 = plot(title="MAE when vs high MDi cows", titlefontsize=10, legend=:outerright)
p7 = plot(title="MPE when vs mid-high MDi cows", titlefontsize=10, legend=:outerright) 
p8 = plot(title="MPE when vs high MDi cows", titlefontsize=10, legend=:outerright)
for i in unique([d₁; d₂])
    lw = i < 2 ? 2 : 1
    la = i < 2 ? 1 : 0.5
    if i in d₁ && count(==(i), n_days₁) > 1
        plot!(p1, ash(coefs₁["status (unhealthy)"][n_days₁ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
        plot!(p3, ash(coefs₁["group (sick)"][n_days₁ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
        plot!(p5, ash(maes₁[n_days₁ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
        plot!(p7, ash(mpes₁[n_days₁ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
    end
    if i in d₂ && count(==(i), n_days₂) > 1
        plot!(p2, ash(coefs₂["status (unhealthy)"][n_days₂ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
        plot!(p4, ash(coefs₂["group (sick)"][n_days₂ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
        plot!(p6, ash(maes₂[n_days₂ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
        plot!(p8, ash(mpes₂[n_days₂ .== i], m=50), hist=false, label="k=$i", xtickfontsize=6, lw=lw, la=la)
    end
end
p9 = bar(d₁, c₁, orientation=:h, legend=false, title="Best k", yticks=d₁)
p0 = bar(d₂, c₂, orientation=:h, legend=false, title="Best k", yticks=d₂)
px = plot(p1, p3, p5, p7, p9, layout=@layout [[a;b;c;d] e])
py = plot(p2, p4, p6, p8, p0, layout=@layout [[a;b;c;d] e])
savefig(px, "mdi1_estimate.png")
savefig(py, "mdi2_estimate.png")

## ----- Find confidence intervals for β₄ and β₅ -----
@info "Running: Confidence Intervals"
struct ConfidenceIntervals{T<:AbstractFloat}
    μ::T
    σ::T
    percent90::NamedTuple{(:lower, :upper), Tuple{T,T}}
    percent95::NamedTuple{(:lower, :upper), Tuple{T,T}}
    percent99::NamedTuple{(:lower, :upper), Tuple{T,T}}
end

function ConfidenceIntervals(μ::T, σ::T, dist::Type{S} = Normal) where 
                            {T<:AbstractFloat, S<:Distribution}
    D = dist(μ, σ)
    μₜ = mean(D)            # Transformed mean
    σₜ = std(D)             # Transformed standard deviation
    percent90 = (lower = quantile(D, 0.05), upper = quantile(D, 0.95))
    percent95 = (lower = quantile(D, 0.025), upper = quantile(D, 0.975))
    percent99 = (lower = quantile(D, 0.005), upper = quantile(D, 0.995))
    return ConfidenceIntervals(μₜ, σₜ, percent90, percent95, percent99)
end

μ₄₁ = mean(coefs₁["status (unhealthy)"][n_days₁ .== 1]); μ₅₁ = mean(coefs₁["group (sick)"][n_days₁ .== 1])
σ₄₁ = std(coefs₁["status (unhealthy)"][n_days₁ .== 1]); σ₅₁ = std(coefs₁["group (sick)"][n_days₁ .== 1])
μ₄₂ = mean(coefs₂["status (unhealthy)"][n_days₂ .== 1]); μ₅₂ = mean(coefs₂["group (sick)"][n_days₂ .== 1])
σ₄₂ = std(coefs₂["status (unhealthy)"][n_days₂ .== 1]); σ₅₂ = std(coefs₂["group (sick)"][n_days₂ .== 1])
CI₄₁ = ConfidenceIntervals(μ₄₁, σ₄₁)
CI₅₁ = ConfidenceIntervals(μ₅₁, σ₅₁)
CI₄₂ = ConfidenceIntervals(μ₄₂, σ₄₂)
CI₅₂ = ConfidenceIntervals(μ₅₂, σ₅₂)
# === How β₄ and β₅ predicts difference in milk yield between healthy vs sick cows ===
# --- β₄ and β₅ when MDi threshold = 1.4 ---
CI1₁ = ConfidenceIntervals(μ₅₁, σ₅₁, LogNormal)                     # healthy cows in sick vs healthy group   [99%: (0.964, 0.969)]
CI2₁ = ConfidenceIntervals(μ₄₁, σ₄₁, LogNormal)                     # healthy vs unhealthy in sick group      [99%: (0.880, 0.887)]
CI3₁ = ConfidenceIntervals(μ₄₁ + μ₅₁, √(σ₄₁^2 + σ₅₁^2), LogNormal)  # healthy in healthy vs unhealthy in sick [99%: (0.850, 0.858)]
# --- β₄ and β₅ when MDi threshold = 1.8 ---
CI1₂ = ConfidenceIntervals(μ₅₂, σ₅₂, LogNormal)                     # healthy cows in sick vs healthy group   [99%: (0.971, 0.977)]
CI2₂ = ConfidenceIntervals(μ₄₂, σ₄₂, LogNormal)                     # healthy vs unhealthy in sick group      [99%: (0.877, 0.888)]
CI3₂ = ConfidenceIntervals(μ₄₂ + μ₅₂, √(σ₄₂^2 + σ₅₂^2), LogNormal)  # healthy in healthy vs unhealthy in sick [99%: (0.853, 0.866)]

## ----- Model fit without accounting cow status -----
fm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum + status + group + normalized_GPTAM)
mae_train₁, mae_test₁, coefs₁, model₁ = categorize_and_fit(train_healthy₁, train_sick₁, 1, :mdi, mdi_threshold₁, fm=fm, split_by=:random, train_size=9, test_size=1, random_state=1234)
mae_train₂, mae_test₂, coefs₂, model₂ = categorize_and_fit(train_healthy₁, train_sick₂, 1, :mdi, mdi_threshold₂, fm=fm, split_by=:random, train_size=9, test_size=1, random_state=1234)
# Model predictions for average cows
dfₕ = DataFrame(dinmilk = repeat(1:200,3), lactnum = repeat([1,2,3],inner=200), group="healthy", status="healthy", normalized_GPTAM=0)
dfₛ = DataFrame(dinmilk = repeat(1:200,3), lactnum = repeat([1,2,3],inner=200), group="sick", status="unhealthy", normalized_GPTAM=0)
dfₕ[!,:yield₁] = predict(model₁, dfₕ) |> x -> exp.(x)
dfₛ[!,:yield₁] = predict(model₁, dfₛ) |> x -> exp.(x)
dfₕ[!,:yield₂] = predict(model₂, dfₕ) |> x -> exp.(x)
dfₛ[!,:yield₂] = predict(model₂, dfₛ) |> x -> exp.(x)
# Build plots
p1 = plot(1:200, dfₕ.yield₁[dfₕ.lactnum .== 1], label="Lactation 1, Healthy", lw=2, lc=:blue, ls=:solid, title="MDi=$mdi_threshold₁", legend=:bottomright, ylims=(0,150), xlabel="Days in milk", ylabel="Yield")
plot!(p1, 1:200, dfₕ.yield₁[dfₕ.lactnum .== 2], label="Lactation 2, Healthy", lw=2, lc=:blue, ls=:dash)
plot!(p1, 1:200, dfₕ.yield₁[dfₕ.lactnum .== 3], label="Lactation 3, Healthy", lw=2, lc=:blue, ls=:dot)
plot!(p1, 1:200, dfₛ.yield₁[dfₛ.lactnum .== 1], label="Lactation 1, Sick", lw=2, lc=:red, ls=:solid)
plot!(p1, 1:200, dfₛ.yield₁[dfₛ.lactnum .== 2], label="Lactation 2, Sick", lw=2, lc=:red, ls=:dash)
plot!(p1, 1:200, dfₛ.yield₁[dfₛ.lactnum .== 3], label="Lactation 3, Sick", lw=2, lc=:red, ls=:dot)

p2 = plot(1:200, dfₕ.yield₂[dfₕ.lactnum .== 1], label="Lactation 1, Healthy", lw=2, lc=:blue, ls=:solid, title="MDi=$mdi_threshold₂", legend=:bottomright, ylims=(0,150), xlabel="Days in milk", ylabel="Yield")
plot!(p2, 1:200, dfₕ.yield₂[dfₕ.lactnum .== 2], label="Lactation 2, Healthy", lw=2, lc=:blue, ls=:dash)
plot!(p2, 1:200, dfₕ.yield₂[dfₕ.lactnum .== 3], label="Lactation 3, Healthy", lw=2, lc=:blue, ls=:dot)
plot!(p2, 1:200, dfₛ.yield₂[dfₛ.lactnum .== 1], label="Lactation 1, Sick", lw=2, lc=:red, ls=:solid)
plot!(p2, 1:200, dfₛ.yield₂[dfₛ.lactnum .== 2], label="Lactation 2, Sick", lw=2, lc=:red, ls=:dash)
plot!(p2, 1:200, dfₛ.yield₂[dfₛ.lactnum .== 3], label="Lactation 3, Sick", lw=2, lc=:red, ls=:dot)
p3 = plot(p1,p2, layout=(1,2))
savefig(p3, "yield_curve.png")

## ----- Average difference in yield between sick vs non-sick -----
@info "Running: Yield Differences" 
list₁.coef["overall"] = list₁.coef["status (unhealthy)"] + list₁.coef["group (sick)"]
list₂.coef["overall"] = list₂.coef["status (unhealthy)"] + list₂.coef["group (sick)"]
p1 = plot(dinmilk_range, exp.(list₁.coef["group (sick)"]), 
          label=L"\frac{healthy_{sick}}{healthy_{healthy}}", 
          title="Yield comparison when MDi threshold=$mdi_threshold₁",
          ylabel="sick/healthy group ratio", xlabel="k days after event",
          fillcolor=:blue, fillrange=0.7, fillalpha=0.2, ylims=(0.7,1.05), legend=:outerright)
plot!(p1, dinmilk_range, exp.(list₁.coef["overall"]), 
      label=L"\frac{unhealthy_{sick}}{healthy_{healthy}}",
      fillcolor=:red, fillrange=0.7, fillalpha=0.2)
p2 = plot(dinmilk_range, exp.(list₂.coef["group (sick)"]), 
          label=L"\frac{healthy_{sick}}{healthy_{healthy}}", 
          title="Yield comparison when MDi threshold=$mdi_threshold₂",
          ylabel="sick/healthy group ratio", xlabel="n days after event",
          fillcolor=:blue, fillrange=0.7, fillalpha=0.2, ylims=(0.7,1.05), legend=:outerright)
plot!(p2, dinmilk_range, exp.(list₂.coef["overall"]), 
      label=L"\frac{unhealthy_{sick}}{healthy_{healthy}}",
      fillcolor=:red, fillrange=0.7, fillalpha=0.2)
p3 = plot(p1, p2, layout = (2,1))
savefig(p3, "yield_compare.png")

## --- Refit data on train set and make prediction on test set ---
@info "Test prediction"
fm = @formula(logyield ~ 1 + log(dinmilk) + dinmilk + lactnum + status + group + normalized_GPTAM)
mae_train₁, mae_test₁, coefs₁, model₁ = categorize_and_fit(train_healthy₁, train_sick₁, 1, :mdi, mdi_threshold₁, fm=fm, split_by=:random, train_size=8, test_size=2, random_state=1234)
mae_train₂, mae_test₂, coefs₂, model₂ = categorize_and_fit(train_healthy₁, train_sick₂, 1, :mdi, mdi_threshold₁, fm=fm, split_by=:random, train_size=8, test_size=2, random_state=1234)