library(jsonlite)
library(data.table)
library(gtools)


detail <- fromJSON("detail.json", flatten=TRUE)
detail <- lapply(detail, function(x) {
  x[sapply(x, is.null)] <- NA
  unlist(x)
})

temp = c()
for(i in 1:length(detail)) {
  row = t(data.frame(detail[i]))
  temp = unique(c(temp,colnames(row)))
}
temp = temp[-grep("data.screenshots", temp)]
temp = temp[-grep("data.movies", temp)]
temp = temp[-grep("data.package", temp)]
temp = temp[-grep("data.dlc", temp)]
temp = temp[-grep("data.developers", temp)]
temp = temp[-grep("data.achievements", temp)]
temp = temp[-grep("requirements.minimum", temp)]

pb <- txtProgressBar(min =1, max = length(detail), style=3)
final = data.frame(matrix(NA, nrow=58039, ncol=115))
colnames(final) = temp

for (i in 1:length(detail)) {
  setTxtProgressBar(pb, i)
  row = t(data.frame(detail[i]))
  row = data.frame(row)
  row = row[,colnames(row) %in% temp]
  for (x in colnames(row)) {
    final[i,x] = toString(row[1,x])
  }
}

final = final[final$data.type == "game",]
final = final[!is.na(final$success),]

write.csv(final, file = "Game_detail.csv",row.names=FALSE)
