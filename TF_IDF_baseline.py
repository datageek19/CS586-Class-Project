from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import num2words
import math
import re

D=[]
porter_stemmer = PorterStemmer()

#To tokenize and to convert inot lower case,removal of stop words.
def word_tokenize_all(list):
	punctuations=['+',',','?','!','/','@','#','$']
	tokens=[]
	for lines in list:
		all_words=word_tokenize(lines)	
		all_words=map(lambda x:x.lower(),all_words)
		filtered_words = [word for word in all_words if word not in stopwords.words('english')+punctuations]
		stemmed_word=[porter_stemmer.stem(word) for word in filtered_words]
		add_numbers=[num2words.num2words(int(words)) if words.isnumeric() else words  for words in stemmed_word]
		tokens.append(add_numbers)
	return tokens

fp=open('./data/train.pair_tok.tsv','r')
list=[line.replace(',', ' ') for line in fp.readlines()]
D=word_tokenize_all(list)
fp.close()

def score(q, d, op_type):
	if op_type=='and':	
		if len(set(q).intersection(d))==len(set(q)):
			ret = 0
			for t in q:
				ret += (tf(t, d) * idf(t))
			return ret
		else:
			return 0
	else:
		ret = 0
		for t in q:
			ret += (tf(t, d) * idf(t))
		return ret
					
def tf(t, d):
	return float(d.count(t)) / len(d)

def idf(t):
	count_in_doucment = 0
	for d in D:
		if t in d:
			count_in_doucment += 1						
	return 0 if count_in_doucment == 0 else math.log(len(D)/float(count_in_doucment))
	
while True:
	#dictionary containing top k documents which matches the query term
	scores={}
	query=input('enter the query string: or q to quit\n')
	print ('**************************************************************\n')
	#checking if the query starts with double quotes(exact match)
	if query.startswith('"'):
		op='and'	
	else:
		op='or'	
		query=re.sub(r'boundary','boundary 4 6',query)
		query=re.sub(r'couple','couple 2',query)
	query=re.sub(r'[#+?!"/@$,]',' ',query)
	if query=='q' or query=='quit':
		break
	query=word_tokenize_all([query])
	i=0
	for d in D:
		#calculating score for each document
		sc=score(query[0],d,op)	
		#storing score in the dictionary only if the score > 0
		if sc != 0:
			scores[i]=sc			
		i+=1
	if len(scores)==0:
		print ('no match found')
	else:
		#getting top 5 documents matching the query
		top_docs=sorted(scores, key=scores.get, reverse=True)[:5]
		rank=1
		for id in top_docs:
			print( "rank:",rank,'score',scores[id])
			print (list[id].strip())
			print( "-----------------------------------------------")
			rank+=1