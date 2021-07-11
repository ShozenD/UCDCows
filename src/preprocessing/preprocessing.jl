using Gadfly: ismissing, isnothing
using CSV: propertynames, SentinelArrays
using CSV, DataFrames, Glob, Dates

cd("src/preprocessing")
include("parsers.jl")

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

#-------------------- Cleaning raw files --------------------
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

#-------------------- Combine files and remove duplicaltes --------------------
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

#-------------------- Save to file --------------------
CSV.write("../../data/analytical/cows-analytic.csv", df)