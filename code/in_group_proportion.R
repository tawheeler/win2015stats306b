library("clusterRepro")

C = as.matrix(read.csv("centroids6.csv"))
rownames(C) <- letters[1:nrow(C)]


mydata = read.csv("all_data_imputed2.csv") # read csv file
mydata$directory <- NULL
mydata$objectnumber <- NULL
D = t(as.matrix(mydata))
rownames(D) <- c("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
                "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "a2", "b2", "c2", "d2", "e2", "f2",
                "g2", "h2", "i2", "j2", "k2", "l2", "m2", "n2", "o2", "p2")

Result <- clusterRepro(C, D, Number.of.permutations = 10)