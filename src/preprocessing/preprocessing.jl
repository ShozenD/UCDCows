using CSV: propertynames, SentinelArrays
using CSV, DataFrames, Glob, Dates

### Cache the files that have already been processed ###
"""
	update_cache(file_paths::Vector{String})

Creates a cache file to record the data files that has already been processed.
If no cache exists, it will create a new cache file.
If a cache file exists it will update it with new file names.

# Examples
```julia
using CSV, DataFrames, Glob

fpaths = glob("*")
update_cache(fpaths)
```

**See also:** `iacwt`
"""
function update_cache(file_paths::Vector{String})
  if !("cache.csv" in glob("*"))
    cache = DataFrame(fname = file_paths)
    CSV.write("cache.csv", cache)
  else # Update cache with new files
    cache = CSV.read("cache.csv", DataFrame)
    fnew = file_paths[[!(fn in cache.fname) for fn in file_paths]]
    append!(cache, DataFrame(fname = fnew))
    CSV.write("cache.csv", cache)
  end
  return cache
end

fpaths = glob("../../data/raw/*")
cache = update_cache(fpaths)

### Renaming Columns ###
colnames = Dict(
  "Animal Number" => "id",
  "Group Number" => "groupid",
  "Days In Milk" => "dinmilk",
  "Lactation Number" => "lactnum",
  "Date" => "date",
  "Begin Time" => "tbegin",
  "End Time" => "tend",
  "Milk duration (mm:ss)" => "milkdur",
  "Yield" => "yield",
  "Yield LF" => "yieldlf",
  "Yield LR" => "yieldlr",
  "Yield RF" => "yieldrf",
  "Yield RR" => "yieldrr",
  "Interval" => "interval",
  "Last Milking Interval" => "last_milk_int",
  "Conductivity RR" => "condrr",
  "Conductivity RF" => "condrf",
  "Conductivity LR" => "condlr",
  "Conductivity LF" => "condlf",
  "Total Conductivity" => "condtot",
  "Mean Flow LR" => "flowlr",
  "Mean Flow LF" => "flowlf",
  "Mean Flow RR" => "flowrr",
  "Mean Flow RF" => "flowrf",
  "Peak Flow LF" => "peaklf", 
  "Peak Flow LR" => "peaklr", 
  "Peak Flow RF" => "peakrf", 
  "Peak Flow RR" => "peakrr", 
  "MDi" => "mdi", 
  "Kickoff LF" => "kicklf", 
  "Kickoff LR" => "kicklr", 
  "Kickoff RF" => "kickrf", 
  "Kickoff RR" => "kickrr", 
  "Blood RF" => "bloodrf", 
  "Blood RR" => "bloodrr", 
  "Blood LF" => "bloodlf", 
  "Blood LR" => "bloodlr", 
  "Total Blood" => "bloodtot", 
  "Teats Not Found" => "teats_not_found", 
  "Smart Pulsation Ratio" => "spr", 
  "Incomplete" => "incomplete"
)

### Parseing Time ### 
function parse_milkdur(x::AbstractArray)
  if typeof(x[1]) == Time
    x = Dates.format.(x, "HH:MM:SS")
    return Time.(x, "MM:SS:ss")
  else
    return Time.(x, "MM:SS") 
  end
end

parse_date(x::AbstractArray) = Date.(x, "m/d/y")
function parse_tbegin(x::AbstractArray)
  x = DateTime.(x, "m/d/y H:MM p")
  return Dates.Time.(x)
end 
parse_tend(x::AbstractArray) = Time.(x, "H:MM p")

function parse_all!(d::DataFrame)
  rename!(d, colnames)

  for col = propertynames(d)
    if col == :date d.date = parse_date(d.date) end
    if col == :tbegin d.tbegin = parse_tbegin(d.tbegin) end
    if col == :tend d.tend = parse_tend(d.tend) end
    if col == :milkdur d.tend = parse_milkdur(d.milkdur) end
  end

  d.tend = d.tbegin + Dates.Minute.(d.tend) + Dates.Second.(d.tend)
end

### Cleaning Loop ### 
fpaths = glob("../../data/raw/*")
cache = update_cache(fpaths)

for path in cache.fname
  # Read data 
  df = CSV.read(path, DataFrame)

  parse_all!(df)

  fname = replace(path, "QMPS Daily Milkings Report_" => "")
  fname = replace(fname, "raw" => "cleaned")
  fname = replace(fname, " " => "")

  CSV.write(string("../../data/cleaned/", fname), df)
end

cleaned_file_paths = glob("../../data/cleaned/*")

df = CSV.read(cleaned_file_paths[1], DataFrame)
for path in cleaned_file_paths[2:end]
  # Read data 
  df1 = CSV.read(path, DataFrame)

  # Append data
  df = [df;df1]
end
sort!(df, [:id, :date, :tbegin])
unique!(df, [:id, :date, :tbegin])

CSV.write("../../data/analytical/cows-analytic.csv", df)

