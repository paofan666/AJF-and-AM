from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import javalang
import re
from javalang import tree
import csv
import os
import pandas as pd

#去除java文件中的注释
def remove_comments(code):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
    def replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)
    return regex.sub(replacer, code)

#节点预处理,处理所有的java文件组合成节点序列
def node_deal(file_path):
    try:
        node_name = []
        fd = open(file_path, 'r', encoding='UTF-8')
        f = fd.read()
        f = remove_comments(f)
        java_tree = javalang.parse.parse(f)
        #类别声明
        for path, node in java_tree.filter(javalang.tree.ClassDeclaration):
            node_name.append(type(node).__name__)
            for i in range(len(node_name)):
                if node_name[i] == "ClassDeclaration":
                    node_name[i] = java_tree.package.name + '.' + node.name
        #接口声明
        for path, node in java_tree.filter(javalang.tree.InterfaceDeclaration):
            node_name.append(type(node).__name__)
            for i in range(len(node_name)):
                if node_name[i] == 'InterfaceDeclaration':
                    node_name[i] = java_tree.package.name + '.' + node.name
        #批注申明
        for path, node in java_tree.filter(javalang.tree.AnnotationDeclaration):
            node_name.append(type(node).__name__)
            for i in range(len(node_name)):
                if node_name[i] == "AnnotationDeclaration":
                    node_name[i] = java_tree.package.name + '.' + node.name
        #枚举声明
        for path, node in java_tree.filter(javalang.tree.EnumDeclaration):
            node_name.append(type(node).__name__)
            for i in range(len(node_name)):
                if node_name[i] == "EnumDeclaration":
                    node_name[i] = java_tree.package.name + '.' + node.name
        return node_name
    except javalang.parser.JavaSyntaxError:
        print(file_path)
        pass
'''处理不能打开的文件    
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



#对PROMISE数据集进行处理，读取csv表格文件,并生成组合文件
with open('D:/Code/java/lucene_data/lucene-2.0.csv', newline='') as input_csvfile:
    reader = csv.reader(input_csvfile)
    with open('D:/Code/java/lucene_data/train_composite_data_20ast.csv', 'w', newline='') as out_csvfile:
        writer = csv.writer(out_csvfile)
        for row in reader:
            data_name, data_bug = row[0], row[21]
            writer.writerow([data_name, data_bug])


#CBOW模型
model = Word2Vec(
    LineSentence(open('D:/Code/java/type_node.txt', 'r', encoding='utf-8')),
    sg=0,
    vector_size=20,
    window=2,
    min_count=1,)
#词向量保存
model.save('word2vec.model')
dic = model.wv.index_to_key
#print(dic)
#print(model.wv[dic[0]])


with open('D:/Code/java/lucene_data/train_composite_data_20ast.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
    #print(data)
    # 特定处理，删除某一特定行x
    #del data[74]
    for row in data:
        #print(row[2])
        # 将csv文件的缺陷处理成0、1的形式
        if row[1] == 'bug':
            pass
        elif row[1] == '0':
            pass
        else:
            row[1] = 1
            with open('D:/Code/java/lucene_data/train_composite_data_20ast.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)


#将词向量的每个维度一次写入csv文件中
df = pd.read_csv('D:/Code/java/lucene_data/train_composite_data_20ast.csv')
new_columns = ['vector' + str(i) for i in range(1, 21)]
df = df.reindex(columns=[*df.columns.tolist(), *new_columns])
for i in range(len(df)):
    word = df.iloc[i, 0]
    vector = model.wv[word]
    for j in range(20):
        df.iloc[i, j+2] = vector[j]
df.to_csv('D:/Code/java/lucene_data/train_composite_data_20ast.csv', index=False)





