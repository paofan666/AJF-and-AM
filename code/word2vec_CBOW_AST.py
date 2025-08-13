from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import javalang
import re
from javalang import tree
import csv
import os
import pandas as pd

#Remove comments from java files
def remove_comments(code):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
    def replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)
    return regex.sub(replacer, code)

#Node preprocessing, processing all java files into node sequences
def node_deal(file_path):
    try:
        node_name = []
        fd = open(file_path, 'r', encoding='UTF-8')
        f = fd.read()
        f = remove_comments(f)
        java_tree = javalang.parse.parse(f)
        #Category Statement
        for path, node in java_tree.filter(javalang.tree.ClassDeclaration):
            node_name.append(type(node).__name__)
            for i in range(len(node_name)):
                if node_name[i] == "ClassDeclaration":
                    node_name[i] = java_tree.package.name + '.' + node.name
        #interface declaration
        for path, node in java_tree.filter(javalang.tree.InterfaceDeclaration):
            node_name.append(type(node).__name__)
            for i in range(len(node_name)):
                if node_name[i] == 'InterfaceDeclaration':
                    node_name[i] = java_tree.package.name + '.' + node.name
        #Annotations and affirmations
        for path, node in java_tree.filter(javalang.tree.AnnotationDeclaration):
            node_name.append(type(node).__name__)
            for i in range(len(node_name)):
                if node_name[i] == "AnnotationDeclaration":
                    node_name[i] = java_tree.package.name + '.' + node.name
        #Enumerate statements
        for path, node in java_tree.filter(javalang.tree.EnumDeclaration):
            node_name.append(type(node).__name__)
            for i in range(len(node_name)):
                if node_name[i] == "EnumDeclaration":
                    node_name[i] = java_tree.package.name + '.' + node.name
        return node_name
    except javalang.parser.JavaSyntaxError:
        print(file_path)
        pass
'''Handle files that can't be opened    
try:
except UnicodeDecodeError:
        print(file_path)
        pass
'''

for foldername, subfolders, filenames in os.walk('D:/Code/PROMISE/lucene-solr-releases-lucene-2.0.0/lucene-solr-releases-lucene-2.0.0/src/java/org/apache/lucene'):
    for filename in filenames:
        if filename.endswith('.java'):
            file_path = os.path.join(foldername, filename)
            fw = open('D:/Code/java/type_node.txt', 'a', encoding='UTF-8')
            for i in node_deal(file_path):
                print(i, file=fw)
            fw.close()



#The PROMISE dataset is processed, the csv table file is read, and the combination file is generated
with open('D:/Code/java/lucene_data/lucene-2.0.csv', newline='') as input_csvfile:
    reader = csv.reader(input_csvfile)
    with open('D:/Code/java/lucene_data/train_composite_data_20ast.csv', 'w', newline='') as out_csvfile:
        writer = csv.writer(out_csvfile)
        for row in reader:
            data_name, data_bug = row[0], row[21]
            writer.writerow([data_name, data_bug])


#CBOW model
model = Word2Vec(
    LineSentence(open('D:/Code/java/type_node.txt', 'r', encoding='utf-8')),
    sg=0,
    vector_size=20,
    window=2,
    min_count=1,)
#Word vector preservation
model.save('word2vec.model')
dic = model.wv.index_to_key
#print(dic)
#print(model.wv[dic[0]])


with open('D:/Code/java/lucene_data/train_composite_data_20ast.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
    #print(data)
    # Specific processing, delete a specific line x
    #del data[74]
    for row in data:
        #print(row[2])
        # Treat defects in the csv file as 0 and 1
        if row[1] == 'bug':
            pass
        elif row[1] == '0':
            pass
        else:
            row[1] = 1
            with open('D:/Code/java/lucene_data/train_composite_data_20ast.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)


#Write each dimension of the word vector to the csv file at once
df = pd.read_csv('D:/Code/java/lucene_data/train_composite_data_20ast.csv')
new_columns = ['vector' + str(i) for i in range(1, 21)]
df = df.reindex(columns=[*df.columns.tolist(), *new_columns])
for i in range(len(df)):
    word = df.iloc[i, 0]
    vector = model.wv[word]
    for j in range(20):
        df.iloc[i, j+2] = vector[j]
df.to_csv('D:/Code/java/lucene_data/train_composite_data_20ast.csv', index=False)





