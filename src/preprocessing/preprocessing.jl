## =========================================================================================
# This files processes all the daily/weekly data imported from the Diamond and Jones
# servers. To run this file, ensure that the `data/` folder contains the following
# subfolders:
#   - `raw/diamond h`
#   - `raw/jones vms`
#   - `raw/jones v300`
#   - `cleaned/diamond h`
#   - `cleaned/jones vms`
#   - `cleaned/jones v300`
# ==========================================================================================

using Gadfly: ismissing, isnothing
using CSV: propertynames, SentinelArrays
using CSV, DataFrames, DataFramesMeta, Glob, Dates

cd(@__DIR__)
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
"""
function update_cache(file_paths::Vector{String})
  if !("cache.csv" in glob("*"))
    cache = DataFrame(fname = file_paths)
    CSV.write("cache.csv", cache)
  else # Update cache with new files
    cache = CSV.read("cache.csv", DataFrame)
    fnew = file_paths[[!(fn in cache.fname) for fn in file_paths]]
    fnew = replace(fnew, "\\" => "/") # for both MacOS and Windows compatibility
    append!(cache, DataFrame(fname = fnew))
    CSV.write("cache.csv", cache)
  end
  return cache
end

#-------------------- Cleaning raw files ---------------------------------------
servers = ["diamond h", "jones v300", "jones vms"]

for server in servers
    fpaths = glob("../../data/raw/$server/*")
    cache = update_cache(fpaths)
    for path in cache.fname
      # Read data
      df = CSV.read(path, DataFrame)
      try
        if occursin("Grain", path)
          parse_grain!(df, path)
    
          fname = replace(path, "Grain Consumed in VMS_" => "grain_")
          fname = replace(fname, "raw" => "cleaned")
          fname = replace(fname, r"(.)+[\\|/]" => "")
        else
          parse_all!(df)
    
          fname = replace(path, "QMPS Daily Milkings Report_" => "milk_")
          fname = replace(fname, "raw" => "cleaned")
          fname = replace(fname, r"(.)+[\\|/]" => "")
        end
        CSV.write(string("../../data/cleaned/$server/", fname), df)
      catch
        println("Failed to parse: ", path)
      end 
    end
end

#-------------------- Combine files and remove duplicates ----------------------
grain_df = DataFrame()
milk_df = DataFrame()
for server in servers
    cleaned_file_paths = glob("../../data/cleaned/$server/*")
    for path in cleaned_file_paths
      # Read data 
      temp = CSV.read(path, DataFrame)
      temp[!,:farm] .= server
      # Append data
      if occursin("grain", path)
        global grain_df = vcat(grain_df, temp)
      else
        global milk_df = vcat(milk_df, temp)
      end
    end
end
sort!(milk_df, [:id, :date, :tbegin])
unique!(milk_df, [:id, :date, :tbegin])

#-------------------- Save to file ---------------------------------------------
CSV.write("../../data/analytical/grain-analytic.csv", grain_df)
CSV.write("../../data/analytical/cows-analytic.csv", milk_df)

