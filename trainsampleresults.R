samplemodelresults<-function(trainsamp, trainPCsamp){

    bagsnum<- 10
    resultslist<-list()
    #Start using parallel here
    library(doParallel)
    registerDoParallel(cores = 5)
    start.time <- Sys.time()
    modFit<- train(classe~ .,data=trainsamp,
                   method="rpart",
                   trControl=trainControl(method = "cv", number = 10))
    end.time <- Sys.time()
    time.taken <- difftime(end.time, start.time, units = "mins")
    print(time.taken)  
    resultslist<-c(resultslist, 
                   cart = modFit$results[modFit$results$Accuracy==max(modFit$results$Accuracy),], cart.time = time.taken)
    
    start.time <- Sys.time()
    modFit<- train(classe~ .,data=trainPCsamp,
                   method="rpart",
                   trControl=trainControl(method = "cv", number = 10))
    end.time <- Sys.time()
    time.taken <- difftime(end.time, start.time, units = "mins")
    print(time.taken)  
    resultslist<-c(resultslist, 
                   cartPC = modFit$results[modFit$results$Accuracy==max(modFit$results$Accuracy),], cartPC.time = time.taken)
    start.time <- Sys.time()
    modFit<- train(classe~ .,data=trainsamp,
                   method="C5.0Tree",
                   trControl=trainControl(method = "cv", number = 10))
    end.time <- Sys.time()
    time.taken <- difftime(end.time, start.time, units = "mins")
    print(time.taken)  
    resultslist<-c(resultslist, c50 = modFit$results, c50.time = time.taken)
    start.time <- Sys.time()
    modFit<- train(classe~ .,data=trainPCsamp,
                   method="C5.0Tree",
                   trControl=trainControl(method = "cv", number = 10))
    end.time <- Sys.time()
    time.taken <- difftime(end.time, start.time, units = "mins")
    print(time.taken)  
    resultslist<-c(resultslist, c50PC = modFit$results, c50PC.time = time.taken)
    start.time <- Sys.time()
    modFit <- train(classe~ .,data=trainsamp,
                    method="treebag", trControl=trainControl(method = "cv", number = 10))
    end.time <- Sys.time()
    time.taken <- difftime(end.time, start.time, units = "mins")
    print(time.taken) 
    resultslist<-c(resultslist, treebag = modFit$results, treebag.time = time.taken)
    start.time <- Sys.time()
    modFit <- train(classe~ .,data=trainPCsamp,
                    method="treebag", trControl=trainControl(method = "cv", number = 10))
    end.time <- Sys.time()
    time.taken <- difftime(end.time, start.time, units = "mins")
    time.taken 
    resultslist<-c(resultslist, treebagPC = modFit$results, treebagPC.time = time.taken)
    
    ##bagged C5.0 decision tree
    c50treeBag <- list()
    c50treeBag$fit<-function(x,y,...)
    {
      library(C50)
      data<-as.data.frame(x)
      data$y <- y
      C5.0(y~.,data = data)
    }
    c50treeBag$pred<-function (object, x) 
    {
      library(C50)
      if (!is.data.frame(x)) x <- as.data.frame(x)
      out <- C50::predict.C5.0(object, x, type = "prob")
      out
    }
    
    c50treeBag$aggregate<-ctreeBag$aggregate
    
    start.time <- Sys.time()
    modFit<- train(x =trainsamp[,-c(54)],
                   y =trainsamp[,c(54)], method = "bag",
                   B=bagsnum,
                   bagControl = bagControl(fit = c50treeBag$fit,
                                           predict = c50treeBag$pred,
                                           aggregate = c50treeBag$aggregate
                   ),
                   trControl=trainControl(method = "cv", number = 10)
    )
    end.time <- Sys.time()
    time.taken <- difftime(end.time, start.time, units = "mins")
    print(time.taken) 
    resultslist<-c(resultslist, c50treebag = modFit$results, c50treebag.time = time.taken)
    
    start.time <- Sys.time()
    modFit<- train(x =trainPCsamp[,-c(45)],
                   y =trainPCsamp[,c(45)], method = "bag",
                   B=bagsnum,
                   bagControl = bagControl(fit = c50treeBag$fit,
                                           predict = c50treeBag$pred,
                                           aggregate = c50treeBag$aggregate
                   ),
                   trControl=trainControl(method = "cv", number = 10)
    )
    end.time <- Sys.time()
    time.taken <- difftime(end.time, start.time, units = "mins")
    print(time.taken) 
    resultslist<-c(resultslist, c50treebagPC = modFit$results, c50treebagPC.time = time.taken)
    
    
    resultlistdf<-as.data.frame(unlist(resultslist))
    names(resultlistdf)<-c("Value")
    resultlistdf$Value<-round(resultlistdf$Value,3)
    names<-read.table(text = rownames(resultlistdf),sep = ".", colClasses = "character")
    dfres<-cbind(names,resultlistdf$Value)
    names(dfres)<-c("Model", "Stat", "Value")
    dfresw<-dcast(dfres, Model ~ Stat, value.var = "Value")
    return(dfresw)
}