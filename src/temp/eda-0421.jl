using 
  CSV, 
  DataFrames, 
  DataFramesMeta, 
  CategoricalArrays,
  Dates,
  Statistics,
  GLM, 
  Gadfly,
  Colors

df = CSV.read("data/daily-milking-example.csv", DataFrame)

colnames = Dict(
  "Animal Number" => "id", 
  "Group Number" => "group", 
  "Lactation Number" => "lactnum",
  "Days In Milk" => "DinMilk",
)
rename!(df, colnames)
df.lactnum = CategoricalArray(df.lactnum)
levels!(df.lactnum, [1,2,3,4,5,6])

# Number of cows
length(unique(df.id))

# Number of data point per cow
gdf = groupby(df, :id)
sdf = combine(gdf, nrow => :obs)
describe(sdf.obs)

# Number of data point per lactation number
gdf = groupby(df, :lactnum)
sdf = combine(gdf, nrow => :obs)
describe(sdf.obs)

# Estimate lacation curve
df.lnDinMilk = log.(df.DinMilk)
df.lnYield = log.(df.Yield .+ 2.718281828459045)

"""
  lact_curve_coefs()

Estimates the coefficients of the lactation curve OLS
"""
function lactcurve_coefs(df::AbstractDataFrame, lnum::Integer=0)
  @assert lnum ∈ [0,1,2,3,4,5,6]

  if lnum != 0
    df = @where(df, :lactnum .== lnum)
  end
  df = @select(df, :lnYield, :lnDinMilk, :DinMilk)
  dropmissing!(df)
  
  ols = lm(@formula(lnYield ~ lnDinMilk + DinMilk), df, dropcollinear=false)

  lna, b, negc = coef(ols)
  a, c = exp(lna), -negc

  return a, b, c
end

"""
  lactcure(x, a, b, c,)

Calculates the estimated lactation curve given a vector of days and estimated coefficents
"""
function lactcurve(x::AbstractVector{T}, a::Number, b::Number, c::Number) where T <: Number
  return a * x .^ b .* exp.(-c * x)
end

N = 500                                   # Number of days
x = collect(1:N)

# Lactation Overall 
a₀, b₀, c₀ = lactcurve_coefs(df, 0)
y₀ = lactcurve(x, a₀, b₀, c₀)

# Lactation 1
a₁, b₁, c₁ = lactcurve_coefs(df, 1)
y₁ = lactcurve(x, a₁, b₁, c₁)

# Lactation 2
a₂, b₂, c₂ = lactcurve_coefs(df, 2)
y₂ = lactcurve(x, a₂, b₂, c₂)

# Lactation 3
a₃, b₃, c₃ = lactcurve_coefs(df, 3)
y₃ = lactcurve(x, a₃, b₃, c₃)

# Lactation 4
a₄, b₄, c₄ = lactcurve_coefs(df, 4)
y₄ = lactcurve(x, a₄, b₄, c₄)

# Lactation 5
a₅, b₅, c₅ = lactcurve_coefs(df, 5)
y₅ = lactcurve(x, a₅, b₅, c₅)

# Lactation 6
a₆, b₆, c₆ = lactcurve_coefs(df, 6)
y₆ = lactcurve(x, a₆, b₆, c₆)

p = Gadfly.plot(
  df,
  x = :DinMilk,
  y = :Yield,
  color = :lactnum,
  Geom.point,
  size=[1pt],
  alpha=[0.7],
  Scale.color_discrete_manual(
    "#444e86", "#955196", "#dd5182", "#ff6e54", "#003f5c", "#ffa600",
  ),
  Scale.y_continuous(minvalue=0, maxvalue=100),
  layer(x=x, y=y₀, Geom.line, color=[colorant"gray"], style(line_width=.7mm)),
  layer(x=x, y=y₂, Geom.line, color=[colorant"#444e86"], style(line_width=.7mm)),
  layer(x=x, y=y₃, Geom.line, color=[colorant"#955196"], style(line_width=.7mm)),
  layer(x=x, y=y₄, Geom.line, color=[colorant"#dd5182"], style(line_width=.7mm)),
  Scale.y_continuous(minvalue=0, maxvalue=100),
  Theme(discrete_highlight_color=c->nothing)
)

img = SVG("scatter.svg")
draw(img, p)

## Expected Yield
colnames = Dict(
  "Expected Yield/Hour LF" => "ExpYieldLF", 
  "Expected Yield/Hour LR" => "ExpYieldLR",
  "Expected Yield/Hour RF" => "ExpYieldRF",
  "Expected Yield/Hour RR" => "ExpYieldRR"
)
rename!(df, colnames)

df.ExpYield = df.ExpYieldLF + df.ExpYieldLR + df.ExpYieldRF + df.ExpYieldRR

p = Gadfly.plot(
  df,
  x = :DinMilk,
  y = :ExpYield,
  color = :lactnum,
  Geom.point,
  size=[1pt],
  alpha=[0.7],
  Scale.color_discrete_manual(
    "#444e86", "#955196", "#dd5182", "#ff6e54", "#003f5c", "#ffa600",
  ),
  Theme(discrete_highlight_color=c->nothing)
)

## Conductivity
rename!(df, Dict("Total Conductivity" => "TotalConductivity"))

# Histogram
Gadfly.plot(
  df,
  x = :TotalConductivity,
  Geom.histogram
)

# Scatter plot with Yield
Gadfly.plot(
  df,
  x = :Yield,
  y = :TotalConductivity,
  Geom.point,
  size=[1pt],
  alpha=[0.7],
  Theme(discrete_highlight_color=c->nothing)
)

## Duration
rename!(df, Dict("Milk duration (mm:ss)" => "MilkDuration"))
df.MilkDurationMin = Dates.Minute.(df.MilkDuration)
df.MilkDurationMin = convert.(Int64, df.MilkDurationMin)

Gadfly.plot(
  df,
  x = :MilkDurationMin,
  Geom.histogram
)



