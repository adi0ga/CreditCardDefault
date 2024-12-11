#Change the directory in lines 25 and 307 before running
#The code have a high runtime and at times,especially during training SVM it may appear to be stuck(But it just takes 20-30 mins to execute)
Models={} #variable to store Reuslts for each model without SMOTE
ModelsSMOTE={}#variable to store Reuslts for each model with SMOTE
#importing necessary libraries and classifiers from scikit learn
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import scikitplot as skplot
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#importing the Dataset for all classifiers except Naive Bayes
#Some modifications has to be done for Naive Bayes which will be done later
data=pd.read_csv("D:\\Intro to Stats\\Project\\Data.csv")#Change the directory before running
ccard=data.copy()
X = ccard.drop('default payment next month',axis=1)#Separating the Defaulter Status From the input parameters
X=X.drop('ID',axis=1)#Removing Unnecessary Columns
y = ccard['default payment next month']#Defaulter Vector
tX,tsX,ty,tsy=model_selection.train_test_split(X,y,test_size=0.2,random_state=57)#Splitting the Data    
#Standardizing the dataset
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
tX = scX.fit_transform( tX )
tsX = scX.transform(tsX )

#Initializing all Clasifiers To be used with necessary initial conditions
clf={"Logistic Regression":linear_model.LogisticRegression(fit_intercept=True,max_iter=50,solver="newton-cholesky"),
     "K-Nearest Neighbours":KNeighborsClassifier(n_neighbors=20),
     "Linear Discriminant Analysis":LinearDiscriminantAnalysis(),
     "Decision Tree Classifier":DecisionTreeClassifier(ccp_alpha=0.0005),
     "MultiLayer Perceptron":MLPClassifier(random_state=57,solver='sgd',activation="logistic"),
     "Support Vector Machine":SVC(probability=(True)),
     "Random Forest":RandomForestClassifier(ccp_alpha=0.0005),
     "Naive Bayes":GaussianNB()
     }
###We can use such a for loop to determine the best value of ccp_alpha for Decision Tree Classifier.The for loop should be run after importing the dataset
###In this case we have ccp_alpha=0.0005

'''
ut=[]
oy=[]


for i in range(0,10000):
    clf_ct=DecisionTreeClassifier(ccp_alpha=(i+1)/10000)
    clf_ct.fit(tX,ty)
    ut.append(clf_ct.score(tX,ty))
    oy.append(clf_ct.score(tsX,tsy))
'''
###A similar loop can also be run to determine the value of k for KNN
'''
k1=[]
k2=[]

for i in range(5,100):
    clf_knn=KNeighborsClassifier(n_neighbors=i)
    clf_knn.fit(tX,ty)
    k1.append(clf_knn.score(tX,ty))
    k2.append(clf_knn.score(tsX,tsy))
'''

def sort_smooth(tsX,tsy,n,pred_prob,stril,verbose,lik):
    """
    # Function for Sorting Smoothing Method.The function serves 2 purposes.
    #First it applies the standard sorting Smoothing method
    #Second when passed with n=0 it returns the PLot of R-squared for the linear regression line of SSM Method for a model vs the value of n
    #tsX and tsY are the testing data variables 
    #n is the No. of neighbourhood points of a given datapoint to be considered
    #stril is a String for the name of Model in plots
    #verbose can take values 0 or 1 indivcating whether SSM is carried out without or with SMOTE respectively
    #lik is variable to make necessary changes for different  datasets(i.e. the one for Naive Bayes and the one For other classifiers)
    Caution:With n=0, the Running time increases significantly
    """
    if lik==0:
        nj=23
    else:
        nj=13
    #separating cases depending on values of n
    if n<0:
        ValueError
    elif n>0:
        #SSM method
        predprob=pred_prob[range(0,len(tsy)),1]#Extracting the probability Vector indicating that a given datapoint is a defaulter
        import pandas as pd
        df1=pd.DataFrame(tsX)
        #making a new copy of vector of predicted probabilities
        if lik==0:
            TsY=tsy.values
        else:
            TsY=tsy
        df1.insert(nj,"Predicted Probabilities",predprob,True)#Including the predicted probailities in the dataset
        df1.insert(nj+1,"tsy",TsY,True)#Including the defaulter vector in the dataset
        df1=df1.sort_values(by=["Predicted Probabilities"],ignore_index=True)#Soring the dataset in ascending order of predicted probabilites
        real_prob=[]
        lop=[]
        #The following for loop Computes the Estimated Real Probabilies
        for i in range(0,len(df1)):
            P1=df1["tsy"][i]
            for j in range(1,50+1):
                if ((i-j)>=0)&((i+j)<=len(tsy)-1):
                    P1=P1+df1["tsy"][i-j]+df1["tsy"][i+j]
                elif (i-j)<0:
                    P1=P1+df1["tsy"][i+j]
                elif ((i+j)>(len(tsy)-1)):
                    P1=P1+df1["tsy"][i-j]
            real_prob.append(P1/(101))
            lop.append(df1["Predicted Probabilities"][i])#indexing Predicted Probabilities corresponding to the calculated Estimated real probability
        list(lop),real_prob
        lop=np.reshape(lop,(-1,1))
        #Linear Regression and Calculation of R-squared
        from sklearn.linear_model import LinearRegression
        lgr=LinearRegression()
        lgr.fit(lop,real_prob)
        rsq=lgr.score(lop,real_prob)
        icpt=lgr.intercept_
        slp=lgr.coef_
        #Scatter Plot for SSM
        plt.scatter(lop,real_prob,marker='o',c=["#FFFFFF"],edgecolor=["#000000"])
        x1=np.linspace(0,1,1000)
        y1=[]
        for i in range(0,1000):
            t=slp*x1[i]+icpt
            y1.append(t)
        #Plotting the Linear Regression Line
        plt.plot(x1,y1,color='#000000')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Estimated Actual Probability")
        lij=[" ","SMOTE"]
        plt.title(label="ScatterPlot for "+stril+lij[verbose]+" for SSM")
        if icpt>=0:
            plt.figtext(0.15,0.7,"Y="+str(round(float(slp),3))+"x+"+str(round(float(icpt),3))+"\n$R^2$="+str(round(float(rsq),5)))
        else:
            plt.figtext(0.15,0.7,"Y="+str(round(float(slp),3))+"x"+str(round(float(icpt),3))+"\n$R^2$="+str(round(float(rsq),5)))
        plt.show()
        return rsq,slp,icpt
    elif n==0:
        #Now Preparing a loop to run SSM over All values of n,from n=5 to n=200 with a step of 5,and record the values of R-squared
        
        lisrsq=[]
        
        for n in range(5,201,5):
            #Usual SSM as above
            predprob=pred_prob[range(0,len(tsy)),1]
            import pandas as pd
            df1=pd.DataFrame(tsX)
            if lik==0:
                TsY=tsy.values
            else:
                TsY=tsy
            df1.insert(nj,"Predicted Probabilities",predprob,True)
            df1.insert(nj+1,"tsy",TsY,True)
            df1=df1.sort_values(by=["Predicted Probabilities"],ignore_index=True)
            real_prob=[]
            lop=[]
            for i in range(0,len(df1)):
                P1=df1["tsy"][i]
                for j in range(1,n+1):
                    if ((i-j)>=0)&((i+j)<=len(tsy)-1):
                        P1=P1+df1["tsy"][i-j]+df1["tsy"][i+j]
                    elif (i-j)<0:
                        P1=P1+df1["tsy"][i+j]
                    elif ((i+j)>(len(tsy)-1)):
                        P1=P1+df1["tsy"][i-j]
                real_prob.append(P1/(2*n+1))
                lop.append(df1["Predicted Probabilities"][i])
            list(lop),real_prob
            lop=np.reshape(lop,(-1,1))
            from sklearn.linear_model import LinearRegression
            lgr=LinearRegression()
            lgr.fit(lop,real_prob)
            rsq=lgr.score(lop,real_prob)
            lisrsq.append(rsq)
            
        #Plotting the results Obtained
        plt.plot(range(5,201,5),lisrsq)
        lij=[" ","SMOTE"]
        plt.xlabel("Value Of n for "+stril+"Model"+lij[verbose])
        plt.ylabel("R-Squared for Linear Regression")
        plt.show()
#Function to index outputs in the results Variables(Models and ModelsSMOTE)
def index(model,tr_accu,ts_accu,tr_ar,ts_as,ssm,verbose):
    if (verbose):
        ModelsSMOTE[model]={"Train accuracy":tr_accu,"Test accuracy":ts_accu,"Train Area Ratio":tr_ar,"Test Area Ratio":ts_as,"SSM:R^2,Slope,intercept":ssm}
    else:
        Models[model]={"Train accuracy":tr_accu,"Test accuracy":ts_accu,"Train Area Ratio":tr_ar,"Test Area Ratio":ts_as,"SSM:R^2,Slope,intercept":ssm}
#We have used a standard library function and modified it to fit our purpose So we have included its Documentation
def plot_cumulative_gain(y_true, y_probas,k,leng,name,
                         ax=None, figsize=None, title_fontsize="large",
                         text_fontsize="medium"):
    """Generates the Cumulative Gains Plot from labels and scores/probabilities
    The cumulative gains chart is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://mlwiki.org/index.php/Cumulative_Gain_Chart. The implementation
    here works only for binary classification.
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.
        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
        title (string, optional): Title of the generated plot. Defaults to
            "Cumulative Gains Curve".
        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
    Example:
        >>> import scikitplot as skplt
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> skplt.metrics.plot_cumulative_gain(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()
        .. image:: _static/examples/plot_cumulative_gain.png
           :align: center
           :alt: Cumulative Gains Plot
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Cumulative Gains for data with '
                         '{} category/ies'.format(len(classes)))

    # Compute Cumulative Gain Curves
    
    percentages, gains =skplot.helpers.cumulative_gain_curve(y_true, y_probas[:, 1],
                                                classes[1])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title('Lift Curve of {}'.format(name), fontsize=title_fontsize)

    ax.plot(leng*percentages,k*gains, lw=3, label='Model')

    ax.set_xlim([0,leng])
    ax.set_ylim([0,math.ceil((k+k/8)/100)*100])

    ax.plot([0, leng], [0, k], 'k--', lw=2, label='Baseline')

    ax.set_xlabel('Number of Total Data', fontsize=text_fontsize)
    ax.set_ylabel('Cumulative Number of Target Data', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=text_fontsize,frameon=False)
    uo=[leng*percentages,k*gains]
    return ax,uo
#The following Function Plots the best fit Curve on The Cumulative Gains chart which the above function Does not do
def plot_best_fit(vect):
    x1=np.linspace(0,len(vect),1000)
    y1=[]
    for i in range(0,1000):
        if  i/1000<=sum(vect)/len(vect):
            y1.append(x1[i])
        else:
            y1.append(sum(vect))
    plt.plot(x1,y1,label="Best Fit Curve")
    plt.legend(frameon=False,loc='lower right')
    return [x1,y1]

def area_ratio(c1,c2,vect):
    #This function computes the area ratio
    d1=c1
    d2=c2
    a1=auc(d1[0],d1[1])#Compute area under model curve
    a2=auc(d2[0],d2[1])#Compute area under best fit
    a2=a2-(1/2)*len(vect)*sum(vect)#Compute area between best fit and baseline curve
    a1=a1-(1/2)*len(vect)*sum(vect)#Compute area between model and baseline curve
    return (a1/a2)
#__main__

#SMOTE TEST TRAIN DATA initialization
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=571)
tX_sm,ty_sm=sm.fit_resample(tX,ty.ravel())
for i in clf:
    if i=="Naive Bayes":
        """
        Naive Bayes Classifier has a strong assumption of independence of parameters,
        But clearly Parameters like Bill Statement for a datapoint and payment amount have heavy dependence
        So we reimport data to modify it to meet the requirements of Naive Bayes Classifier
        """
        file=pd.read_csv("D:\\Intro to Stats\\Project\\Data.csv")#Change the directory before running
        db=file.copy()
        vlp=[]
        clp=[]
        output='default payment next month'
        #The following for loop sums up all the Bill statements into a one parameter and Payment amount into another parameter.
        #This modification to the data has been done for Naive Bayes after discussion with Prasun De,it is because Naive Bayes has
        #strong independence assumption so summing up all the bill and pay amounts increases its accuracy and overall performance
        for j in range(0,30000):
            s=0
            t=0
            for it in range(1,7):
                s=s+file["BILL_AMT"+str(it)][j]
                t=t+file["PAY_AMT"+str(it)][j]
            vlp.append(s)
            clp.append(t)
        file.insert(21,"BILL_AMT",vlp)#Inserting the New parameters
        file.insert(22,"PAY_AMT",clp)#Inserting the New parameters
        #Deleteing the monthly payment and bill statemnet parameters
        for it in range(1,7):
            del file["BILL_AMT"+str(it)]
            del file["PAY_AMT"+str(it)]
        cols = [ f for f in file.columns if file.dtypes[ f ] != "object"]
        #Removing output vector and other unused parameters
        cols.remove( "ID")
        cols.remove( output )
        X1=file[cols].values#Final dataset after Modification
        y1=file[output].values#Defaulter Vector
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split( X1, y1, test_size=0.2,random_state=(5))
        #Standardizing
        scX = StandardScaler()
        X_train = scX.fit_transform( X_train )
        X_test = scX.transform( X_test )
        sm=SMOTE(random_state=51090)
        X_train_sm,y_train_sm=sm.fit_resample(X_train,y_train.ravel())
        #Standardizing
        from sklearn.preprocessing import StandardScaler
        scX = StandardScaler()
        X_train_sm = scX.fit_transform( X_train_sm )
    for k in range(0,2):
        #Initializing different variables for different Classifiers
        if (k==0)&(i!="Naive Bayes"):
            clf[i].fit(tX,ty)#fitting the classifier to the training data set
            j=" "
            tX1=tX
            ty1=ty
            tsX1=tsX
            tsy1=tsy
            lik=0
        elif(k==1)&(i!="Naive Bayes"):
            clf[i].fit(tX_sm,ty_sm)#fitting the classifier to the training data set
            j=" SMOTE "
            tX1=tX_sm
            ty1=ty_sm
            tsX1=tsX
            tsy1=tsy
            lik=0
        elif (k==0)&(i=="Naive Bayes"):
            clf[i].fit(X_train,y_train)#fitting the classifier to the training data set
            j=" "
            tX1=X_train
            ty1=y_train
            tsX1=X_test
            tsy1=y_test
            lik=1
        else:
            clf[i].fit(X_train_sm,y_train_sm)#fitting the classifier to the training data set
            j=" SMOTE "
            tX1=X_train_sm
            ty1=y_train_sm
            tsX1=X_test
            tsy1=y_test
            lik=1

        print("Results For "+i+" Model ")
        print("The test accuracy of the "+i+j+" Model is:",clf[i].score(tsX1,tsy1))
        print("The train accuracy of the "+i+j+" Model is:",clf[i].score(tX1,ty1))
        y_pred1=clf[i].predict(tsX1)#Prediction for testing data
        y_pred=clf[i].predict(tX1)#Prediction for testing data
        pred_prob=clf[i].predict_proba(tsX1)#Predicted Probabilities for a datapoint in the testing dataset to be a defaulter
        mk,lm=plot_cumulative_gain(tsy1,pred_prob,sum(tsy1),len(tsy1),i+"(Testing Data)"+j)#Plotting the lift Curve for testing Data
        lm1=plot_best_fit(tsy1)#Plotting best fit curve on the Lift Curve
        ar1=(area_ratio(lm, lm1,tsy1))#Area Ratio for Testing Data
        pred_prob1=clf[i].predict_proba(tX1)#Predicted Probabilities for a datapoint in the training dataset to be a defaulter
        mk,lm=plot_cumulative_gain(ty1,pred_prob1,sum(ty1),len(ty1),i+"(Training Data)"+j)#Plotting the lift Curve for testing Data
        lm1=plot_best_fit(ty1)#Plotting best fit curve on the Lift Curve
        plt.show()
        ar0=(area_ratio(lm, lm1,ty1))#Area Ratio for Training Data
        ssm=sort_smooth(tsX1, tsy1,50,pred_prob,i,k,lik)#Applying Sorting Smoothing Method On testing Data
        
        #Uncomment the line below to run the loop for n vs R^2 in SSM(This part of code has a very high run time of about 4-5 hrs)
        #ssm=sort_smooth(tsX1, tsy1,0,pred_prob,i,k,lik)
        
        #Indexing the model results to corresponding Variables
        index(i,clf[i].score(tX1,ty1),clf[i].score(tsX1,tsy1),ar0,ar1,ssm,k)
        
        #Printing Classification Reports and Confusion Matrix
        print("Testing Classification Report")
        print(classification_report(tsy1, y_pred1))
        print("\n"+100*"*")
        print("Training Classification Report")
        print(classification_report(ty1, y_pred))
        print("\n"+100*"*")
        print("Testing Confusion Matrix")
        print(confusion_matrix(tsy1, y_pred1))
        print("\n"+100*"*")
        print("Training Confusion Matrix")
        print(confusion_matrix(ty1, y_pred))
        print("\n"+100*"*")