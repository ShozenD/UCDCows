using CSV, DataFrames, DataFramesMeta, Glob, Dates

cd(@__DIR__)
include("parsers.jl")

df = CSV.read("../../data/diagnostic/diagnosis-2.csv", DataFrame)

colset = [:ID,:LACT,:DIM,:Quarter,:Dateok,:Trt_MAST,:Trt_dec,:TRT,:CULT_res,:CULT_type,:Trt_REMAST]
select!(df, colset)

colnames = Dict(
  "ID" => "id",
  "LACT" => "lactnum",
  "DIM" => "dinmilk",
  "Quarter" => "teat",
  "Dateok" => "datehosp",
  "Trt_MAST" => "trtmast",
  "Trt_dec" => "trttype",
  "TRT" => "trtflag",
  "CULT_res" => "cultres",
  "CULT_type" => "culttype",
  "Trt_REMAST" => "trtremast"
)

rename!(df, colnames)

df.datehosp = parse_date(df.datehosp)

function cultflag(x::String)
  if |(x == "NG", x == "NO")
    return 0
  else
    return 1
  end
end

df.cultflag = passmissing(cultflag).(df.culttype)

# Cows that recieved only 1 treatment
df₁ = @where(df, ismissing.(:trtremast))
select!(df₁, Not([:trtremast]))

# Cows that recieved 2 treatments
df₂ = @where(df, ismissing.(:trtremast) .== 0)
select!(df₂, [:id, :datehosp, :trtremast])
rename!(df₂, Dict("datehosp" => "datehosp2"))

df₃ = leftjoin(df₁, df₂, on = :id)
df₃.mastflag = df₃.trtflag .== 1 .| df₃.cultflag .== 1
df₃.mastflag = passmissing(Int).(df₃.mastflag)

select!(
  df₃, 
  [:id, :lactnum, :dinmilk, :datehosp, :teat, :trtflag, :cultflag, :mastflag, :culttype, :datehosp2, :trtremast]
)

df₄ = @where(df₃, ismissing.(:mastflag) .== 0)

CSV.write("../../data/diagnostic/diagnosis-2-clean.csv", df₄)