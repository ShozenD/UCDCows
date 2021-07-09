# hack to save Table as HTML file

using PrettyTables

function savehtml(filename, data)
    open("$filename.html", "w") do f
        pretty_table(f, data, backend = :html)
    end
end

using CSV, DataFrames

df = DataFrame(CSV.File("src/temp/population.csv"))

savehtml("population_original", df)

df.id = 1:nrow(df)

df = select!(df, :id, :)

savehtml("population_working.html", df)

# change headers with a vector
headers = [
  :id, :country, :region, :subregion, :wiki18, :wiki19, :wikipct
]

rename!(df, headers)

df.wiki18nocomma = replace.(df.wiki18, "," => "")
df.wiki19nocomma = replace.(df.wiki19, "," => "")

df.wiki18Int = parse.(Int, df.wiki18nocomma) 
df.wiki19Int = parse.(Int, df.wiki19nocomma)

df.my18 = round.(df.wiki18Int ./ 10^6, digits = 3)
df.my19 = round.(df.wiki19Int ./ 10^6, digits = 3)

savehtml("population_working.html", df)


df.mydelta = round.(df.my19 .- df.my18, digits = 3)
savehtml("population_working.html", df)

df.mypct = round.(
  ((df.wiki19Int .- df.wiki18Int) ./ df.wiki18Int) .* 100, digits = 2
)
savehtml("population_working.html", df)

# construct new df containing selected columns in selected order 
myheaders = [
  :id, :region, :subregion, :country, :my18, :my19, :mydelta, :mypct
]

mydf = select(df, myheaders)

savehtml("mydf", mydf)

mydf_t = mydf[end, :]
mydf_t = DataFrame(mydf_t)

savehtml("mydf_t", mydf_t)

delete!(mydf, nrow(mydf))
savehtml("mydf", mydf)

mydf_d = describe(mydf)
mydf_d = describe(mydf, :all)

regions_u = unique(mydf.region)
vscodedisplay(regions_u)

subregions_u = unique(mydf.subregion)
vscodedisplay(subregions_u)

mydf = sort!(mydf, :country)
savehtml("mydf", mydf)

myorder = [:region, :subregion, :country]
mydf = sort!(mydf, myorder)
savehtml("mydf", mydf)

my18_t = sum(mydf.my18)
my19_t = sum(mydf.my19)

mydelta_t = my19_t - my18_t

mypct_t = mydelta_t / my18_t * 100

# wiki totals
wiki18_t = mydf_t.my18[1]
wiki19_t = mydf_t.my19[1]

wikidelta_t = mydf_t.mydelta[1]

wikipct_t = mydf_t.mypct[1]

mydf.my100 = round.(
  mydf.my19 ./ sum(my19_t) .* 100, digits = 2
)
savehtml("mydf", mydf)

# verify new column add us to 100%

sum(mydf.my100)

# group data by region

mygdf = groupby(mydf, :region)

show(mygdf, allgroups = true)

mygdf_c = combine(mygdf,
  :my18 => sum,
  :my19 => sum,
  :mydelta => sum
)

mygdf_c.mypct_c = round.(
  mygdf_c.mydelta_sum ./ mygdf_c.my18_sum .* 100, digits = 2
)

mygdf_c.my100_c = round.(
  mygdf_c.my19_sum ./ sum(mygdf_c.my19_sum) .* 100, digits = 2
)

savehtml("mygdf_c", mygdf_c)

# check totals 

my18gdf_t = sum(mygdf_c.my18_sum)

# sort to find fastest growing regions by population

# group data by subregion

mygdf_s = groupby(mydf, :subregion)

# show mygdf_s in REPL
show(mygdf_s, allgroups = true)

mygdf_s_c = combine(mygdf_s,
  :region => unique,
  :my18 => sum,
  :my19 => sum,
  :mydelta => sum
)

# add percent columns 
mygdf_s_c.mypct_s = round.(
  mygdf_s_c.mydelta_sum ./ mygdf_s_c.my18_sum .* 100, digits = 2
)

mygdf_s_c.my100_s = round.(
  mygdf_s_c.my19_sum ./ sum(mygdf_s_c.my19_sum) .* 100, digits = 2
)

savehtml("mygdf_s_c", mygdf_s_c)

my18gdf_s_t = sum(mygdf_s_c.my18_sum)

my19gdf_s_t = sum(mygdf_s_c.my19_sum)

CSV.write("mydf.csv", mydf)
CSV.write("mygdf_c.csv", mygdf_c)
CSV.write("mygdf_s_c.csv", mygdf_s_c)

