using DataFrames

df = readtable("all_data.csv")

ind_entropy = findfirst(sym->sym == :entropy, names(df))
df = df[:,[1:ind_entropy-1, ind_entropy+1:end]]

m,n = size(df)
keep = trues(m)
for i = 1 : m
    for j = 1 : n
        if isnan(df[i,j])
            keep[i] = false
            break
        end
    end
end

df2 = df[keep, :]
writetable("all_data_nona.csv", df2)