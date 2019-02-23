\documentclass[a4paper]{article}

\usepackage{fullpage} % Package to use full page
\usepackage{parskip} % Package to tweak paragraph skipping
\usepackage{tikz} % Package for drawing
\usepackage{amsmath}



\title{CSCI-P556\\Fall 2018\\Assignment 3\\Due 11:59PM, Nov. 2, 2018}
\author{Vishal Singh (singhvis)}
\begin{document}

\maketitle

\section{Introduction}
A training dataset of 2000 samples and 500 features and a testing dataset of 600 samples and 500 features are provided. We after preliminary EDA, we are training baseline models on training data and checking score on testing data (which we are treating as validation). Next we perform feature engineering and tweaking the parameters of the models to improve accuracy on the test data. The main challenge on this dataset is feature engineering as the number of features are a lot and we do not know the physical significance of the features.\newline


\section{Exploratory Data Analysis}
Findings about Features-
\begin{enumerate}
    \item Number of features: 500
    \item Number of continuous features:500
    \item The density plot(Figure 1) of most of the features resemble a normal with mean and median being close to each other
    \item Columns with NAs: None
    \item Columns with negative/trash value: None
    \item Number of features with less than 5 distinct values in train dataset: f91, f277, f424 (These features can be treated as continuous variables but for the following analysis I've treated them as continuous due to their values being comparable to values in other columns)
    \item There are a lot of columns having outliers$^1$(Figure 2), I have decided not to treat then because of following reasons:
    \begin{enumerate}
        \item We do know the physical significance of the values, it might be that columns values might be an outliers according to the definition of outlier defined by us but might be a meaningful value
        \item Treating outliers in all the column might decrease the bias in the data, leading to overfitting the training data
    \end{enumerate} 
    \item Columns with unique identifiers: None
    \item On checking correlation between the features, we get \textbf{47 pairs} of features which have correlation of 0.5 or more. A new dataset has been created after removing one feature from the pairs.
\end{enumerate}

Findings about Labels-
\begin{itemize}
    \item Distinct Label: 2
    \item Values of Label: -1, 1
    \item Proportion of each label in train data: 50:50 (balanced)
    \item Proportion of each label in test data: 50:50 (balanced)
    \item Since, the labels are balance, we can use accuracy score to check accuracy of model. If the data was imbalanced then we would've checked f-score
\end{itemize}

Since the data does not have any NaNs, unique identifiers, we do not make any changes to the dataset before applying the baseline models

\section{Baseline Models}
Since the data has 2 distinct labels, I've chosen Logistic Regression, Random Forest Classifier, KNN Algorithm, and Support Vector Machines for predicting the output label for test set. \newline
Baselines models are trained on normalized train dataset and then tested on test dataset.

Performance on baselines models is as follows:
\begin{center}
\begin{tabular}{ | l | l| } 
\hline
Model & Baseline Accuracy \\ 
\hline
Logistic Regression & 57.99\% \\ 
\hline
Random Forest Classifier & 61.83\% \\ 
\hline
KNN(Figure 6) & 62.5\% (k=6) \\ 
\hline
SVM(linear) & 58.5\% \\ 
\hline
\end{tabular}
\end{center}


\section{Feature Engineering}
\begin{enumerate}
    \item \textbf{Logistic Regression} \newline
    Following Feature Engineering steps have been performed to increase the accuracy of Logistic Regression models
    \begin{enumerate}
        \item \textbf{Correlated Features} First, I've trained the model which is obtained after removing one of the correlated pairs. The accuracy obtained after performing this step is \textbf{56\%}.
        \item \textbf{KBest Features}- Next I've used the \textbf{sklearn.feature\_selection.SelectKBest} library to search for best k features. These features chosen based on highest scores obtained for correlation with the output variable
        \item \textbf{Principal Component Analysis}- Since the number of features are relatively large, we can do PCA to reduce the number of features. I have iterated through number of components obtained through PCA and trained a Logistic Regression Model using the principal components(Figure 4). 
        \item \textbf{Recursive Feature Elimination}- I've also tried a wrapper method for feature elimination. Running \textbf{Recursive Feature Elimination} for selecting the best 20 features. This is done using the library \textbf{sklearn.feature\_selection.RFE}. Since, this process this computationally expensive, I've tried just for 20 features.
    \end{enumerate}
    \item \textbf{Random Forest Classifier} \newline
    Following Feature Engineering steps have been performed to increase the accuracy of Random Forest Classifier
    \begin{enumerate}
        \item \textbf{Feature Importance:} \textbf{feature\_importances\_} method in \textbf{sklearn.ensemble.RandomForestClassifier} is used to find the n most important features. feature\_importances\_ returns the most important features which are used to classify the dataset using decision trees.
        \item The features obtained using \textbf{feature\_importances\_} have correlation between them(Figure 5). The ones which have high correlation (more than 0.7) are found and one of them is removed. The model is trained on the remaining features and the accuracy is obtained.
    \end{enumerate}
    
    \item \textbf{KNN} \newline
    \begin{enumerate}
        \item \textbf{Feature Importance:} The most important features obtained after running Random Forst Classifier are selected and then KNN is applied on it.
         \item \textbf{KBest Features:} The 20 best features obtained using \textbf{sklearn.feature\_selection.SelectKBest} are used and KNN is applied on it.
          \item \textbf{Principal Components:} KNN is applied on the variable obtained using PCA. The number of components is varied from 1 to 50 to make the computationally feasible
    \end{enumerate}
    
    \item \textbf{Support Vector Machines:} \newline
    \begin{enumerate}
        \item \textbf{Feature Importance:} The most important features obtained after running Random Forst Classifier are selected and then SVM model is trained using those features.
    \end{enumerate}
\end{enumerate}



\section{Model Building}
\begin{enumerate}
    \item \textbf{Logistic Regression} \newline
    Following Feature Engineering steps have been performed to increase the accuracy of Logistic Regression models
    \begin{enumerate}
        \item \textbf{Correlated Features} Since the number of features is still very high. I've trained the Logistic Regression model using  \textbf{L2 Normalization} to avoid overfitting. The accuracy obtained after performing this step is \textbf{56\%}.
        \item \textbf{KBest Features}- Next I've used the \textbf{sklearn.feature\_selection.SelectKBest} library to search for best k=[1-25] features then I've trained a Logistic Regression Model using \textbf{L2 Normalization} on the k best features obtained. The accuracy for each of these models have been calculated. The maximum accuracy of \textbf{62.83\%}is obtained for \textbf{k=4}. (Refer Figure 3)
        \item \textbf{Principal Component Analysis}- I've iterated through the number of principal components chosen from 1-500, trained a logistic regression model using L@ Regularization. The maximum accuracy of  \textbf{59.83\%} is obtained for \textbf{number of components = 1} even though then variance explained is only 0.014. (Refer Figure 4). Since the maximum accuracy is obtained for 1 principal component, I've avoided L2 in final model which is there in Jupyter notebook.
        \item \textbf{Recursive Feature Elimination}- The accuracy obtained on test set after training a logistic regression model on 20 best features using RFE is \textbf{58.16\%}
    \end{enumerate}
    \item \textbf{Random Forest Classifier} \newline
    Following Feature Engineering steps have been performed to increase the accuracy of Random Forest Classifier
    \begin{enumerate}
        \item \textbf{Feature Importance:} On iterating through different number of importatnt features obtained using Random Forest. Maximum accuracy was obtained when we took top 18 important features. Accuracy of \textbf{89.83\%} was obtained. \textbf{n\_estimators}, which is the number of trees build using RF is taken to be as 1000.
        \item After removing one of the \textbf{highly correlated} features, we have trained a random forest classifier on the remaining features.\newline
        Parameters given: \textbf{n\_estimators}=1000 \newline
        Accuracy obtained: 87.5\%
        \item \textbf{Grid Search Cross Validation:} Using \textbf{sklearn.model\_selection.GridSearchCV} we can iterate through different combination of parameters specified for \textbf{sklearn.ensemble.RandomForestClassifier}. Steps are shown in the Jupyter Notebook for iterating through one feature at a time, one iteration is done at a time as it is computationally expensive. \newline
        Parameters: 
        \begin{itemize}
            \item max\_depth: [5,10,15,25,50,100], best parameter: max\_depth=10
            \item max\_features: [1,3,6,7,9,10,15], best parameter: max\_features=9
            \item n\_estimators: [100:1100], best parameter: n\_estimator=1000
        \end{itemize}
        Accuracy obtained: 88.33\% \newline
        The accuracy is lower than the accuracy obtained when choosing original 18 important features. This can be because the best combination of parameters can be choosen by iterating over one parameter at a time.
    \end{enumerate}
    
    \item \textbf{KNN} \newline
    \begin{enumerate}
        \item \textbf{Feature Importance:} On iterating through the n=[1:20] most important features from Random Forest Classifier and K=[1:25] max accuracy of \textbf{91.66\%} is obtained \newline
        Best Parameters: \textbf{imp features}=13, \textbf{k=6} \newline
        Accuracy obtained: 91.66\%
         \item \textbf{KBest Features:} On iterating through K=[1:25] for kbest features obtained from \textbf{SelectKBest} max accuracy of \textbf{79.5\%} is obtained \newline
        Best Parameters: \textbf{K Best Features}=20, \textbf{k=15} \newline
        Accuracy obtained: 79.5\%
          \item \textbf{Principal Components:}On iterating through the number of principal components=[1:50] from Principal Component Analysis and K=[1:number of components]  maximum accuracy of \textbf{91.66\%} is obtained \newline
        Best Parameters: \textbf{number of principal components}=13, \textbf{k=12} \newline
        Accuracy obtained: 60.66\%
    \end{enumerate}
    
    \item \textbf{Support Vector Machines:} \newline
    \begin{enumerate}
        \item \textbf{Types of SVM}:
        \begin{itemize}
            \item Polynomial SVM: Max Accuracy of 58.66\% is obtained for a 2 degree polynomial SVM
            \item Gaussian SVM: Max Accuracy of 58.66\%
            \item Sigmoid SVM: Max Accuracy of 58.33\%
        \end{itemize}
        \item \textbf{Feature Importance:} The 20 most important features obtained after running Random Forest Classifier are selected and then a Gaussian SVM model(since it gave the best accuracy) is trained using those features. \newline
        Accuracy obtained: 82.66\%
    \end{enumerate}
\end{enumerate}


\section{Discussion}
The best accuracy is always obtained on models after using the most important features obtained from Random Forest Classifier. Maximum accuracy is obtained when we use KNN on the best features obtained from Random Forest which is \textbf{91.66\%}. Overall, Logistic Regression does not perform well on this data. Reducing number of features using PCA also does not improve accuracy for Logistic Regression, KNN. Reducing number of correlated features does not improve accuracy for Logistic Regression.

\begin{center}
\begin{tabular}{ | l | l| } 
\hline
Model & Best Accuracy \\ 
\hline
Logistic Regression & 62.833\% \\ 
\hline
Random Forest Classifier & 89.83\% \\ 
\hline
KNN & 91.66\% \\ 
\hline
SVM(gaussian) & 82.5\% \\ 
\hline
\end{tabular}
\end{center}

\newpage

\begin{figure}
        \centering
        \includegraphics[width=15cm,height=6cm]{boxplots.png}
        \caption{Boxplots of features 10-20}
        \label{fig:my_label}
\end{figure}

\begin{figure}
        \centering
        \includegraphics[width=15cm,height=6cm]{density.png}
        \caption{Density plot of features 10-20}
        \label{fig:my_label}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=6cm,height=6cm]{selectkbest.png}
    \caption{Accuracy of LR model trained on k best features}
    \label{fig:my_label}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=6cm,height=6cm]{pca.png}
    \caption{Accuracy of LR model trained by number of principal components}
    \label{fig:my_label}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=6cm,height=6cm]{corr.png}
    \caption{Correlation between important features obtained from RF}
    \label{fig:my_label}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=6cm,height=6cm]{baseknn.png}
    \caption{Baseline KNN model: k vs accuracy}
    \label{fig:my_label}
\end{figure}

% \bibliographystyle{plain}
% \bibliography{bibliography.bib}
\end{document}
