import numpy as np
import pandas as pd
import string
import csv
import os
import sys
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split




def preprocessing():
	data=pd.read_csv('QTA.csv',encoding='ISO-8859-1')
	data=data.to_numpy()

	
	data=data[data[:,-1]>=1000] #removing tags with score<=3 and tagcounts,1000
	data=data[data[:,0]>3]



	def html(text):    #removing html tags
		clean=re.compile('<.*?>')
		return re.sub(clean,'',text)

	def punct(string): #removing punctuation marks
		punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
		for x in string.lower():
			if x in punctuations:
				string=string.replace(x,"")
		return string

	def fun(str):     #converting to lower case
		return str.lower()
	body=data[:,2]
	title=data[:,1]
	# html tag removals
	body=np.vectorize(html)(body)
	title=np.vectorize(html)(title)
	

	# punctuation removal
	body=np.vectorize(punct)(body)
	title=np.vectorize(punct)(title)
	 
	# small letters and split
	body=np.vectorize(fun)(body)
	title=np.vectorize(fun)(title)


	data[:,2]=body
	data[:,1]=title
	
	def lemmatizer(ls): # lemmatizer and remove stop words
		lem=WordNetLemmatizer()
		stopw=set(stopwords.words('english'))
		wordtokens=word_tokenize(ls)

		tem=[]

		for w in wordtokens:
			if w not in stopw:
				tem.append(lem.lemmatize(w))
		
		return tem


	with open('QTA.csv','w') as file: #QTA with body of question and tags with lemmatized, stop words and html tags removed,
		writer=csv.writer(file)
		writer.writerow(["Qbody","tags"])
		
		for a in range(data.shape[0]):
			title=data[a,1]
			title=lemmatizer(title)
			body=data[a,2]
			body=lemmatizer(body)
			writer.writerow([body,data[a,3]])


			
	






def combineIdsContentTag(Tags):   #making a csv file with tag count in it:: ["id","Tags","tagcount"] tag.csv
	temp=Tags[np.argsort(Tags[:,0])]
	
		

	with open('tag.csv','w') as file:
		writer=csv.writer(file)
		writer.writerow(["id","Tags"])
		marker=0
		content=''
		ref=temp[marker,0]

		for a in range(Tags.shape[0]):
			if (ref==temp[a,0] and (temp[a,1]!='\n' and temp[a,1]!='')) :
				content+=str(temp[a,1])+' '
			else:
				writer.writerow([ref,content])
				ref=temp[a,0]
				content=''
				if (ref==temp[a,0] and (temp[a,1]!='\n' and temp[a,1]!='')) :
					content+=str(temp[a,1])+' '
		writer.writerow([ref,content])


	temp=pd.read_csv('tag.csv',encoding='ISO-8859-1')
	temp=temp.sort_values("Tags")
	temp=temp.to_numpy()


	with open('tag.csv','w') as file:
		writer=csv.writer(file)
		writer.writerow(["id","Tags","tagcount"])
		marker=0
		count=0
		ref=temp[marker,1]

		for a in range(temp.shape[0]):
			if (ref==temp[a,1] ) :
				count+=1
			elif(ref!=temp[a,1]):
				while marker!=a:
					writer.writerow([temp[marker,0],temp[marker,1],count])
					marker+=1
				ref=temp[a,1]
				count=1
		while marker!=temp.shape[0]:
			writer.writerow([temp[marker,0],temp[marker,1],count])
			marker+=1
		





def mergeQTA(Questions,Tags):
	
	
	Questions=np.delete(Questions,1,1)
	Questions=np.delete(Questions,1,1)
	Questions=np.delete(Questions,1,1)
	

	

	combineIdsContentTag(Tags) #making a csv file with tag count in it

	TA=pd.read_csv('tag.csv',encoding='ISO-8859-1')
	TA=TA.to_numpy()
	TA=TA[np.argsort(TA[:,0])]


	with open('QTA.csv','w') as file:  #making a QTA csv file with ["score","title","Qbody","tags","tagcount"]
		writer=csv.writer(file)
		writer.writerow(["score","title","Qbody","tags","tagcount"])
		count=0
		MAX=len(Questions[:,0])
		for a in range(Questions.shape[0]):
			if count!=MAX:
				if(Questions[count,0]==TA[a,0]):
					writer.writerow([Questions[a,1],Questions[a,2],Questions[a,3],TA[count,1],TA[count,2]])
					count+=1

				else:
					print("something is wrong1")
					break
			else:
				print("something is wrong2")
				break
				

def getX(body):
	vectorize=TfidfVectorizer()
	vectors = vectorize.fit_transform(body)
	return vectors

def getY(tag):
	labelencoder = LabelEncoder()
	tag = labelencoder.fit_transform(tag)
	return tag


	
def read_data():	
	q='data/Questions.csv'

	t='data/Tags.csv'


	Questions=pd.read_csv(q,encoding='ISO-8859-1')
	Questions=Questions.to_numpy()
	
	Tags=pd.read_csv(t,encoding='ISO-8859-1')
	Tags=Tags.to_numpy()

	return Questions,Tags

def main():
	Questions,Tags =read_data();#reading data
	
	mergeQTA(Questions,Tags)#making a QTA csv file with ["score","title","Qbody","tags","tagcount"]
	preprocessing()#QTA with body of question and tags with lemmatized, stop words and html tags removed,
	data=pd.read_csv('QTA.csv',encoding='ISO-8859-1')
	data=data.to_numpy()
	body=data[:,0]
	tag=data[:,1]
	X=getX(body) #TFIDF
	y=getY(tag)  #LABEL ENCODER
 
	print("X and Y ready")
	

	X_train_1, X_test_1, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
	print(X_train_1.shape, X_test_1.shape, y_train.shape, y_test.shape)

	
	#ridge classifier
	# from sklearn.linear_model import RidgeClassifier
	# clf=RidgeClassifier().fit(x_train,y_train)
	# res=clf.score(x_test,y_test)
	# print("mean accuracy of ridge classifier:: ",res)
	# print("com")
	

	# ridge classifier
	from sklearn import metrics
	from sklearn.linear_model import RidgeClassifier
	rc1=RidgeClassifier()
	rc1.fit(X_train_1,y_train)
	pred1=rc1.predict(X_test_1)
	print("#Ridge classifier")
	print(metrics.classification_report(y_test, pred1))


	# naive bayes
	from sklearn.naive_bayes import BernoulliNB, MultinomialNB
	rc2=MultinomialNB()
	rc2.fit(X_train_1,y_train)
	pred2=rc2.predict(X_test_1)
	print("#naive bayes")
	print(metrics.classification_report(y_test, pred2))


	# perceptron
	from sklearn.linear_model import Perceptron
	rc3=Perceptron()
	rc3.fit(X_train_1,y_train)
	pred3=rc3.predict(X_test_1)
	print("#perceptron")
	print(metrics.classification_report(y_test, pred3))

	#linear svm
	from sklearn.linear_model import SGDClassifier
	rc4=SGDClassifier(loss='hinge')
	rc4.fit(X_train_1,y_train)
	pred4=rc4.predict(X_test_1)
	print("#linear svm")
	print(metrics.classification_report(y_test, pred4))


	#logistic regression
	rc5=SGDClassifier(loss='log')
	rc5.fit(X_train_1,y_train)
	pred5=rc5.predict(X_test_1)
	print("#logistic regression")
	print(metrics.classification_report(y_test, pred5))

if __name__=='__main__':
	main()