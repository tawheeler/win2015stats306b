
using DataFrames


df_trace = readtable("../trace_features.csv", header=false)
df_image = readtable("../all_image_features.csv")

trace_names = Array(Symbol, size(df_trace,2))
trace_names[1] = :isCell
for i = 2 : length(trace_names)
    trace_names[i] = symbol(@sprintf("trace%2d", i-1))
end

names!(df_trace, trace_names)

# now we need to combine the datasets

m = size(df_image,1)
order_index = Array(Int, m)
for i = 1 : m
    order_index[i] = int(df_image[i, :directory]) * 1000 + int(df_image[i, :objectnumber])
end

p = sortperm(order_index) # sort smallest to highest

df = hcat(df_image[p,:], df_trace)

writetable("../all_data.csv", df)