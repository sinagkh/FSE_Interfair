args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[[1]], .libPaths()))
suppressPackageStartupMessages(library(ScottKnottESD))
set.seed(20260928)
x <- read.csv(args[[2]], stringsAsFactors=FALSE)
x$algebraic <- tolower(as.character(x$algebraic)) == 'true'
results <- list(); errors <- list(); i <- 1; k <- 1
for (family in sort(unique(x$family))) {
  z <- x[x$family==family & !x$algebraic,]
  models <- sort(unique(z$arm)); seeds <- sort(unique(z$seed))
  wide <- matrix(NA_real_,length(seeds),length(models),dimnames=list(as.character(seeds),models))
  for (j in seq_len(nrow(z))) wide[as.character(z$seed[j]),z$arm[j]] <- z$value[j]
  if (length(models)<2 || length(seeds)!=10 || anyNA(wide)) stop(paste('Incomplete family',family))
  fit <- tryCatch(ScottKnottESD::sk_esd(as.data.frame(if(z$direction[1]=='higher') wide else -wide),alpha=.05),error=function(e)e)
  if(inherits(fit,'error')) {errors[[k]]<-data.frame(family=family,error=conditionMessage(fit));k<-k+1;next}
  for (m in names(fit$groups)) {
    results[[i]]<-data.frame(family=family,task=z$task[1],architecture=z$architecture[1],metric=z$metric[1],
      arm=m,n=nrow(wide),direction=z$direction[1],mean=mean(wide[,m]),sd=sd(wide[,m]),
      rank=as.integer(fit$groups[[m]]),package='ScottKnottESD',version=as.character(packageVersion('ScottKnottESD')))
    i<-i+1
  }
}
write.csv(do.call(rbind,results),args[[3]],row.names=FALSE)
write.csv(if(length(errors)) do.call(rbind,errors) else data.frame(family=character(),error=character()),args[[4]],row.names=FALSE)
writeLines(capture.output(sessionInfo()),args[[5]])
if(length(errors)) warning('Undefined ESD families retained in the error export; no ranks imputed.')
