using DataFrames, Gadfly
include("utilities.jl")

"""
  plot_conductivity(D, id[, th_mdi=-1.0])

Plots the conductivity trends for each teat. 
If a MDi threshold is given, the resulting plot would have bands indicating
the time frames where the threshold is exceeded.

**See also:** `plot_conductivity_render`
"""
function plot_conductivity end

"""
  plot_yield(D, id[, th_mdi=-1.0])

Plots the yield trends for each teat. 
If a MDi threshold is given, the resulting plot would have bands indicating
the time frames where the threshold is exceeded.
"""
function plot_yield end

function find_headtail(x::AbstractVector)
  xs = [0;x[1:end-1]] # shifted array with first element equal to zero
  y = x - xs

  start = findall(y .== 1)
  finish = findall(y .== -1)

  if length(start) != length(finish)
    append!(finish, length(x))
  end
  return start, finish
end

### Conductivity ###

function plot_conductivity(D::DataFrame, 
                           id::Int, 
                           thmdi=1.0,
                           smooth=1)

  d = @where(D, :id .== id)
  d.dtbegin = d.date + d.tbegin
  d.dtend = d.date + d.tend
  transform!(d, [:condrr,:condrf,:condlr,:condlf] .=> 
    linterp .=> 
    [:condrr,:condrf,:condlr,:condlf]
  )
  if smooth > 1
    d.condrr = rolling_average(d.condrr, smooth)
    d.condrf = rolling_average(d.condrf, smooth)
    d.condlr = rolling_average(d.condlr, smooth)
    d.condlf = rolling_average(d.condlf, smooth)
  end
  
  d.mdi = Impute.interp(d.mdi) |> Impute.locf()
  if thmdi == 1.0 || sum(d.mdi .> thmdi) == 0
    plot_conductivity_render(d)
  else
    plot_conductivity_render(d, thmdi)
  end 
end

function plot_conductivity_render(d::DataFrame)
  plot(d, x = :dtbegin, y = :condrr, Geom.line, color = [colorant"#003f5c"],
    layer( y = :condrf, Geom.line, color = [colorant"#7a5195"] ),
    layer( y = :condlr, Geom.line, color = [colorant"#ef5675"] ),
    layer( y = :condlf, Geom.line, color = [colorant"#ffa600"] ),
    Guide.manual_color_key("Teat", ["RR","RF","LR","LF"], ["#003f5c","#7a5195","#ef5675","#ffa600"]),
    Guide.xlabel("DateTime"),
    Guide.ylabel("Conductivity"),
    Guide.title("Conductivity Trends"),
    Scale.y_continuous(format = :plain),
    Theme(background_color = "white")
  )
end 

function plot_conductivity_render(d::DataFrame, thmdi::Float64)
  s, e = find_headtail(d.mdi .> thmdi)
  d₂ = DataFrame(Start = d.dtbegin[s], End = d.dtend[e])

  plot(d, x = :dtbegin, y = :condrr, Geom.line, color = [colorant"#003f5c"],
    layer( y = :condrf, Geom.line, color = [colorant"#7a5195"] ),
    layer( y = :condlr, Geom.line, color = [colorant"#ef5675"] ),
    layer( y = :condlf, Geom.line, color = [colorant"#ffa600"] ),
    layer(d₂, xmin=:Start, xmax=:End, Geom.vband, color=[colorant"grey"], alpha=[0.4]),
    Guide.manual_color_key("Teat", ["RR","RF","LR","LF"], ["#003f5c","#7a5195","#ef5675","#ffa600"]),
    Guide.xlabel("DateTime"),
    Guide.ylabel("Conductivity", orientation = :vertical),
    Guide.title("Conductivity Trends"),
    Scale.y_continuous(format = :plain),
    Theme(background_color = "white")
  )
end 

### Yield ### 

function plot_yield(D::DataFrame, 
                    id::Int, 
                    thmdi::Float64=1.0, 
                    smooth::Int=1)

  d = @where(D, :id .== id)
  d.dtbegin = d.date + d.tbegin
  d.dtend = d.date + d.tend
  transform!(d, [:yieldrr,:yieldrf,:yieldlr,:yieldlf] .=> 
    linterp .=> 
    [:yieldrr,:yieldrf,:yieldlr,:yieldlf]
  )
  if smooth > 1
    d.yieldrr = rolling_average(d.yieldrr, smooth)
    d.yieldrf = rolling_average(d.yieldrf, smooth)
    d.yieldlr = rolling_average(d.yieldlr, smooth)
    d.yieldlf = rolling_average(d.yieldlf, smooth)
  end

  d.mdi = Impute.interp(d.mdi) |> Impute.locf()
  if thmdi == 1.0 || sum(d.mdi .> thmdi) == 0
    plot_yield_render(d)
  else
    plot_yield_render(d, thmdi)
  end 
end

function plot_yield_render(d::DataFrame)
  plot(d, x = :dtbegin, y = :yieldrr, Geom.line, color = [colorant"#003f5c"],
    layer( y = :yieldrf, Geom.line, color = [colorant"#7a5195"] ),
    layer( y = :yieldlr, Geom.line, color = [colorant"#ef5675"] ),
    layer( y = :yieldlf, Geom.line, color = [colorant"#ffa600"] ),
    Guide.manual_color_key("Teat", ["RR","RF","LR","LF"], ["#003f5c","#7a5195","#ef5675","#ffa600"]),
    Guide.xlabel("DateTime"),
    Guide.ylabel("Yield"),
    Guide.title("Yield Trends"),
    Scale.y_continuous(format = :plain),
    Theme(background_color = "white")
  )
end

function plot_yield_render(d::DataFrame, thmdi::Float64)
  s, e = find_headtail(d.mdi .> thmdi)
  d₂ = DataFrame(Start = d.dtbegin[s], End = d.dtend[e])

  plot(d, x = :dtbegin, y = :yieldrr, Geom.line, color = [colorant"#003f5c"],
    layer( y = :yieldrf, Geom.line, color = [colorant"#7a5195"] ),
    layer( y = :yieldlr, Geom.line, color = [colorant"#ef5675"] ),
    layer( y = :yieldlf, Geom.line, color = [colorant"#ffa600"] ),
    layer(d₂, xmin=:Start, xmax=:End, Geom.vband, color=[colorant"grey"], alpha=[0.4]),
    Guide.manual_color_key("Teat", ["RR","RF","LR","LF"], ["#003f5c","#7a5195","#ef5675","#ffa600"]),
    Guide.xlabel("DateTime"),
    Guide.ylabel("Yield"),
    Guide.title("Yield Trends"),
    Scale.y_continuous(format = :plain),
    Theme(background_color = "white")
  )
end

# Yield per min
function plot_ypm(D::DataFrame, 
  id::Int, 
  thmdi::Float64=1.0, 
  smooth::Int=1)

d = @where(D, :id .== id)
d.dtbegin = d.date + d.tbegin
d.dtend = d.date + d.tend
transform!(d, [:ypmrr,:ypmrf,:ypmlr,:ypmlf] .=> 
linterp .=> 
[:ypmrr,:ypmrf,:ypmlr,:ypmlf]
)
if smooth > 1
d.ypmrr = rolling_average(d.ypmrr, smooth)
d.ypmrf = rolling_average(d.ypmrf, smooth)
d.ypmlr = rolling_average(d.ypmlr, smooth)
d.ypmlf = rolling_average(d.ypmlf, smooth)
end

d.mdi = Impute.interp(d.mdi) |> Impute.locf()
if thmdi == 1.0 || sum(d.mdi .> thmdi) == 0
plot_ypm_render(d)
else
plot_ypm_render(d, thmdi)
end 
end

function plot_ypm_render(d::DataFrame)
plot(d, x = :dtbegin, y = :ypmrr, Geom.line, color = [colorant"#003f5c"],
layer( y = :ypmrf, Geom.line, color = [colorant"#7a5195"] ),
layer( y = :ypmlr, Geom.line, color = [colorant"#ef5675"] ),
layer( y = :ypmlf, Geom.line, color = [colorant"#ffa600"] ),
Guide.manual_color_key("Teat", ["RR","RF","LR","LF"], ["#003f5c","#7a5195","#ef5675","#ffa600"]),
Guide.xlabel("DateTime"),
Guide.ylabel("ypm"),
Guide.title("ypm Trends"),
Scale.y_continuous(format = :plain),
Theme(background_color = "white")
)
end

function plot_ypm_render(d::DataFrame, thmdi::Float64)
s, e = find_headtail(d.mdi .> thmdi)
d₂ = DataFrame(Start = d.dtbegin[s], End = d.dtend[e])

plot(d, x = :dtbegin, y = :ypmrr, Geom.line, color = [colorant"#003f5c"],
layer( y = :ypmrf, Geom.line, color = [colorant"#7a5195"] ),
layer( y = :ypmlr, Geom.line, color = [colorant"#ef5675"] ),
layer( y = :ypmlf, Geom.line, color = [colorant"#ffa600"] ),
layer(d₂, xmin=:Start, xmax=:End, Geom.vband, color=[colorant"grey"], alpha=[0.4]),
Guide.manual_color_key("Teat", ["RR","RF","LR","LF"], ["#003f5c","#7a5195","#ef5675","#ffa600"]),
Guide.xlabel("DateTime"),
Guide.ylabel("ypm"),
Guide.title("ypm Trends"),
Scale.y_continuous(format = :plain),
Theme(background_color = "white")
)
end

### Prior Warning ###
function yield_priorwarn(D::DataFrame, mdi::Float64, n::Int)
  @assert n >= 1
  @assert mdi > 1

  h, t = find_headtail(D.mdi .> mdi)

  h₁ = h[1]
  if 1 < h₁ & h₁ < n
    n = h₁-1
  end
  ŷ_rr = mean(D.yieldrr[h₁-n:h₁-1])
  ŷ_rf = mean(D.yieldrf[h₁-n:h₁-1])
  ŷ_lr = mean(D.yieldlr[h₁-n:h₁-1])
  ŷ_lf = mean(D.yieldlf[h₁-n:h₁-1])

  return [ŷ_rr, ŷ_rf, ŷ_lr, ŷ_lf]
end

function plot_yieldpriorwarn(D::DataFrame, mdi::Float64, n::Int)
  @assert n >= 1
  y = Vector{Float64}(undef,0)
  N = Vector{Int}(undef,0)
  teat = repeat(["rr", "rf", "lr", "lf"], n)

  D.mdi = linterp(D.mdi)
  h, t = find_headtail(D.mdi .> mdi)
  h₁ = h[1]

  for i in 1:n
    y = [y;yield_priorwarn(D, mdi, i)]
    N = [N;repeat([i],4)]
  end

  data = DataFrame([
    :N => N,
    :teat => teat,
    :yield => y
  ])

  maxcondt = minmax_teat(
    D.condrr[h₁], D.condrf[h₁], D.condlr[h₁], D.condlf[h₁], 
    min=false
  )

  s = [1.5mm]
  plot(data,
    layer(@where(data, :teat .== String(maxcondt)), 
          x=:yield, 
          y=:N, 
          size = s,
          Geom.point, 
          color = [colorant"#ef5675"]
    ),
    layer(@where(data, :teat .!= String(maxcondt)), 
          x=:yield, 
          y=:N, 
          size = s,
          Geom.point, 
          color = [colorant"#488f31"]
    ),
    Guide.yticks(ticks=1:n),
    Guide.xlabel("Average Yield"),
    Guide.ylabel("Days Prior"),
    Theme(background_color = "white")
  )
end