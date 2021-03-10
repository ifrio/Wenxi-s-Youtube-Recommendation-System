#https://themeforest.net/category/cms-themes?sort=sales
#https://domains.google/
import requests
from requests_html import HTMLSession
import json
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
session = HTMLSession()
vectorizer = TfidfVectorizer()

def html(url):
  # get htmldoc and write it into txt
  s = requests.session()
  data = s.get(url)
  with open("youtube.txt","w+",encoding='UTF-8') as f:
    f.write(data.content.decode("utf-8"))
  # read the file
  return data.content.decode("utf-8")

def save(name, data):
  with open(name + ".txt","wb") as f:
    pickle.dump(data,f)

def getlink(url,new = False):
  if new == True:
    # start to collect YouTube links
    links = []
    beginning = '/watch?v='
    v = []

    # the text contains YT video links
    line1 = html(url)

    while beginning in line1:
      a = line1.index(beginning) + len(beginning)
      link = line1[a:a+11]
      if link not in v and link != '':
        v.append(link)

        link = 'https://www.youtube.com/watch?v=' + str(link)

        with open("video list.txt","a") as f:
          f.write("\n"+link)

        links.append(link)
      line1 = line1[a+11:]
  else:
    with open("video list.txt","r") as f:
      links = f.read().split()
  return links

def get_data(link):
  a = '"videoDetails":'
  b = ',"isCrawlable"'
  line = html(link)
  info = line[line.index(a)+len(a):line.index(b)] + '}'
  info = json.loads(info)
  return info

def getUsefulData(info):
  #get data we need
  data = []
  columns = ["title","lengthSeconds","keywords","shortDescription"]
  for col in columns:
    for key in info.keys():
      if col == key:
        data.append(info[col])
  return data

def xdata(link,new= False):   
  xdata = []
  videolinks = getlink(link,new)
  count = 0

  for video in videolinks:
    print(count)
    try:
      info = get_data(video)
      data = getUsefulData(info)
      count = count + 1
      xdata.append(data)
    except ValueError:
      save("x_data",xdata)
  
  save("x_data",xdata)

  return xdata

def loadData():
  X = []
  with open("x_data.txt","rb")as f:
    X = pickle.load(f)
    df = pd.DataFrame(data = X,columns = ["title","lengthSeconds","keywords","shortDescription"])
    return df.dropna()

def experiment_1a():
  data = loadData()
  x = data["shortDescription"]
  y = data.index

  model = svm.SVC()
  model.fit(vectorizer.fit_transform(list(x)), y)

  return model.score(vectorizer.transform(list(data["title"])), data.index)

def experiment_1b():
  data = loadData()
  x = data["title"]
  y = data.index

  model = svm.SVC()
  model.fit(vectorizer.fit_transform(list(x)), y)

  return model.score(vectorizer.transform(list(data["shortDescription"])), data.index)

def experiment_1c():
  data = loadData()
  temp = data[["shortDescription","title"]]
  temp.reset_index(level=0, inplace=True)
  print(temp.columns)
  temp["index"] = temp.index
  temp = temp[["shortDescription","title"]]
  train = temp.applymap(lambda x: x[:int(len(x)/2)])
  test = temp.applymap(lambda x: x[int(len(x)/2):])

  model = svm.SVC()
  model.fit(vectorizer.fit_transform(list(train)), list(data["index"]))

def experiment_2a():
  data = loadData()
  x = data["shortDescription"]
  y = data.index

  model = KNeighborsClassifier(n_neighbors = 1)
  model.fit(vectorizer.fit_transform(list(x)), y)

  return model.score(vectorizer.transform(list(data["title"])), data.index)

def experiment_2b():
  data = loadData()
  x = data["shortDescription"]
  y = data.index

  model = svm.SVC()
  model.fit(vectorizer.fit_transform(list(x)), y)

  return model.score(vectorizer.transform(list(data["title"])), data.index)

def results():
  print("#################### Experiments ####################")
  print(" ")
  print("########## Experiment #1A: Short Desc. Data #########")
  print("Experiment 1a) Accuracy: ", experiment_1a())
  print(" ")
  print("########## Experiment #1B: Title Desc. Data #########")
  print("Experiment 1b) Accuracy: ", experiment_1b())
  print(" ")
  print(" ")
  print("########## Experiment #2A: KNN Classifier ###########")
  print("Experiment 2a) Accuracy: ", experiment_2a())
  print(" ")
  print("########## Experiment #2B: SVM ######################")
  print("Experiment 2b) Accuracy: ", experiment_2b())

@app.route("/recommendVideo/<sentence>")
def recommendVideo(sentence):
  data = loadData()
  x = data["title"]
  y = data.index

  model = svm.SVC()
  model.fit(vectorizer.fit_transform(list(x)), y)
  return str(model.predict(vectorizer.transform([sentence])))

app.run(host="0.0.0.0")