import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import jieba
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier



def multiclass_logloss(actual, predicted, eps=1e-15):
    """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
    :param actual: 包含actual target classes的数组
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
train=pd.read_csv(r"Train_Data.csv")
test=pd.read_csv(r"Test_Data.csv")
sub=pd.read_csv(r"Submit_Example.csv")
del test["negative"]

y=train["negative"]

list_fumian=["头条","京东","花呗","京东白条","京东众筹","返利","360金融","360借条","借钱","支付宝","红岭控股"]
def split_s(s):
    
    list_aa=[]
    #print(s)
    list_b=[]
    list_a=s.split(";")
    list_a = sorted(list_a,key = lambda i:len(i),reverse=True)  
   
    aa="yang"
    for n in range(len(list_a)):
        aa=list_a.pop()
        c=len(list_a)
        for i in list_a:
            
            if aa in i:
                break
            if aa in list_fumian:
                break
            c=c-1
        if c==0 :
            list_b.append(aa)

    list_aa.append(list_b)

    list_aaa=[]
    for i in list_aa:
        s=i[0]
        for j in range(1,len(i)):
            s=s+";"+i[j]
        list_aaa.append(s)
    #print(list_aaa[0])
    return list_aaa[0]


x_y=pd.DataFrame({"entity":train["entity"],"key_entity":train["key_entity"]})
x_y=x_y.dropna()
dict_a=dict(zip(x_y["entity"],x_y["key_entity"]))
#
test["entity"]=test["entity"].fillna("无")

list_entity=[]
count=0
for i in test["entity"]:
    if i in dict_a:
        list_entity.append(dict_a[i])
    else:
        list_entity.append(split_s(i))



del train["negative"],train["key_entity"]




list_b=[]
data=pd.concat((train,test))
#data.index=list(range(9999))
data["entity"]=data["entity"].fillna("无")
data["text"]=data["entity"]+data["text"]
data["text"]=data["text"].fillna("无")

count=0
for i in data["text"]:
    count=count+1
    list_a=jieba.cut(i, cut_all=True)
    list_b.append(" ".join(list_a))

data["jieba"]=list_b

"""停用词"""
#stwlist=[line.strip() for line in open(r'C:\Users\NB\Desktop\word_stop.txt','r',encoding='utf-8').readlines()]
   
def number_normalizer(tokens):
    """ 将所有数字标记映射为一个占位符（Placeholder）。
    对于许多实际应用场景来说，以数字开头的tokens不是很有用，
    但这样tokens的存在也有一定相关性。 通过将所有数字都表示成同一个符号，可以达到降维的目的。
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))
tfv = NumberNormalizingVectorizer(min_df=3,  
                                  max_df=0.8,
                                  max_features=None,                 
                                  ngram_range=(1, 2), 
                                  use_idf=True,
                                  smooth_idf=True,
                              
                                  #stop_words=stwlist
                                 )
#
##构造转换器
tfidf=TfidfVectorizer()

#用tfidf对数据向量化
xtrain_tfv=tfidf.fit_transform(data["jieba"])
xtrain_tf=xtrain_tfv.toarray()
#
#
##分割数据
xtrain_tvf=xtrain_tf[:4999]
xtest_tvf=xtrain_tf[4999:]
train_x,test_x,train_y,test_y=train_test_split(xtrain_tvf,y,test_size=0.2,random_state=2019)
#
#
#
#tor=LogisticRegression()


tor=MultinomialNB()
tor=RandomForestClassifier()
param_dic={}
nb=GridSearchCV(tor,param_grid=param_dic,cv=4)
nb.fit(train_x,train_y)
lr=nb.predict_proba(test_x)

finish=nb.predict(xtest_tvf)

finish_1=list(finish)
scor=lr[:,1]

fpr, tpr, thresholds = metrics.roc_curve(test_y, scor, pos_label = 1)
score=metrics.auc(fpr, tpr)
print(score)
prent=nb.score(test_x,test_y)
print("准确率",prent)

sub["negative"]=finish_1
sub["key_entity"]=np.array(list_entity)


for i in range(5000):
    if sub["key_entity"][i]=="":
        sub["negative"][i] ==0
    if sub["negative"][i]==0:
        
        sub["key_entity"][i]=""
        
sub.to_csv("submision.csv",index=False)
