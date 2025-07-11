#数据处理为20维静态+20维词向量（bug为0/1）的csv文件
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
        for path, node in java_tree.filter(javalang.tree.ClassDeclaration):
            node_name.append(type(node).__name__)
            for i in range(len(node_name)):
                if node_name[i] == "ClassDeclaration":
                    node_name[i] = java_tree.package.name + '.' + node.name
        for path, node in java_tree.filter(javalang.tree.InterfaceDeclaration):
            node_name.append(type(node).__name__)
            for i in range(len(node_name)):
                if node_name[i] == 'InterfaceDeclaration':
                    node_name[i] = java_tree.package.name + '.' + node.name
        # 针对被解析成AnnotationDeclaration的接口
        for path, node in java_tree.filter(javalang.tree.AnnotationDeclaration):
            node_name.append(type(node).__name__)
            for i in range(len(node_name)):
                if node_name[i] == "AnnotationDeclaration":
                    node_name[i] = java_tree.package.name + '.' + node.name
        # 针对枚举类解析问题
        for path, node in java_tree.filter(javalang.tree.EnumDeclaration):
            node_name.append(type(node).__name__)
            for i in range(len(node_name)):
                if node_name[i] == "EnumDeclaration":
                    node_name[i] = java_tree.package.name + '.' + node.name
        return node_name
    except javalang.parser.JavaSyntaxError:
        print(file_path)
        pass


for foldername, subfolders, filenames in os.walk('D:/Code/PROMISE/xalan-j_2_5_0-src/xalan-j_2_5_0/src'):
    for filename in filenames:
        if filename.endswith('.java'):
            file_path = os.path.join(foldername, filename)
            fw = open('D:/Code/java/type_node.txt', 'a', encoding='UTF-8')
            for i in node_deal(file_path):
                print(i, file=fw)
            fw.close()


#对PROMISE数据集进行处理，读取csv表格文件,并生成组合文件
df = pd.read_csv('D:/Code/java/xalan_data/xalan-2.5.csv')
df.drop(df.columns[[0, 1]], axis=1, inplace=True)
df.to_csv('D:/Code/java/xalan_data/test_composite_data.csv', index=False)


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
word_vectors_dict = {word: model.wv[word] for word in model.wv.index_to_key}

#x修改bug类型为0/1
with open('D:/Code/java/xalan_data/test_composite_data.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
    # 特定处理，删除某一特定行x

    #del data[4]

    for row in data:
        #print(row[2])
        # 将csv文件的缺陷处理成0、1的形式
        if row[21] == 'bug':
            pass
        elif row[21] == '0':
            pass
        else:
            row[21] = 1
            with open('D:/Code/java/xalan_data/test_composite_data.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)


df = pd.read_csv('D:/Code/java/xalan_data/test_composite_data.csv')
words = df.iloc[:, 0]
rows_to_delete = []
for i in words:
    if i not in dic:
        df = df[df['name.1'].str.contains(i, na=False) == False]
for i, word in enumerate(words):
    vector = model.wv[word]
    for j, value in enumerate(vector):
        column_name = f'vector_{j + 1}'
        df.at[i, column_name] = value
bug = df['bug']
df.drop(columns=['bug'], inplace=True)
df['bug'] = bug
df.to_csv('D:/Code/java/xalan_data/test_composite_data.csv', index=False)
















