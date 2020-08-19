
**复习：**数据分析的第一步，加载数据我们已经学习完毕了。当数据展现在我们面前的时候，我们所要做的第一步就是认识他，今天我们要学习的就是**了解字段含义以及初步观察数据**。

## 1 第一章：数据载入及初步观察

### 1.4 知道你的数据叫什么
我们学习pandas的基础操作，那么上一节通过pandas加载之后的数据，其数据类型是什么呢？

**开始前导入numpy和pandas**


```python
import numpy as np
import pandas as pd
```

#### 1.4.1 任务一：pandas中有两个数据类型DateFrame和Series，通过查找简单了解他们。然后自己写一个关于这两个数据类型的小例子🌰[开放题]


```python
# Series的字符串表现形式为：索引在左边，值在右边。由于我们没有为数据指定索引，于是会⾃动创 建⼀个0到N-1（N为数据的⻓度）的整数型索引
# 如果只传⼊⼀个字典，则结果Series中的索引就是原字典的键（有序排列）。你可以传⼊排好序的字 典的键以改变顺序
# 对于许多应⽤⽽⾔，Series最重要的⼀个功能是，它会根据运算的索引标签⾃动对⻬数据

# DataFrame是⼀个表格型的数据结构，它含有⼀组有序的列，每列可以是不同的值类型（数值、字符 串、布尔值等）
# 建DataFrame的办法有很多，最常⽤的⼀种是直接传⼊⼀个由等⻓列表或NumPy数组组成的字典
# 如果指定了列序列，则DataFrame的列就会按照指定顺序进⾏排列
```


```python
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
example_1 = pd.Series(sdata)
example_1
```




    Ohio      35000
    Texas     71000
    Oregon    16000
    Utah       5000
    dtype: int64




```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
example_2 = pd.DataFrame(data)
example_2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ohio</td>
      <td>2000</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ohio</td>
      <td>2001</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ohio</td>
      <td>2002</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nevada</td>
      <td>2001</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nevada</td>
      <td>2002</td>
      <td>2.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nevada</td>
      <td>2003</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
example_3 = pd.DataFrame({'Population': [35000, 71000, 16000, 5000]},
                        index=['Ohio', 'Texas', 'Oregon', 'Utah'])
example_3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>35000</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>71000</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>16000</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>5000</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.4.2 任务二：根据上节课的方法载入"train.csv"文件



```python
df = pd.read_csv('C:\\Users\\hanhoo\\train_chinese.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>乘客id</th>
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



也可以加载上一节课保存的"train_chinese.csv"文件。通过翻译版train_chinese.csv熟悉了这个数据集，然后我们对trian.csv来进行操作
#### 1.4.3 任务三：查看DataFrame数据的每列的项


```python
df.columns
```




    Index(['乘客id', '是否幸存', '仓位等级', '姓名', '性别', '年龄', '兄弟姐妹个数', '父母子女个数', '船票信息',
           '票价', '客舱', '登船港口'],
          dtype='object')



#### 1.4.4任务四：查看"cabin"这列的所有项 [有多种方法]


```python
df['客舱'].head()
```




    0     NaN
    1     C85
    2     NaN
    3    C123
    4     NaN
    Name: 客舱, dtype: object




```python
# 不用添加字符串引号

df.客舱.head()
```




    0     NaN
    1     C85
    2     NaN
    3    C123
    4     NaN
    Name: 客舱, dtype: object



#### 1.4.5 任务五：加载文件"test_1.csv"，然后对比"train.csv"，看看有哪些多出的列，然后将多出的列删除
经过我们的观察发现一个测试集test_1.csv有一列是多余的，我们需要将这个多余的列删去


```python
test_1 = pd.read_csv('test_1.csv')
test_1.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>




```python
del test_1['a']
test_1.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



【思考】还有其他的删除多余的列的方式吗？


```python
'''

方法一：直接del DF['column-name'] 

方法二：采用drop方法，有下面三种等价的表达式：

1. DF= DF.drop('column_name', 1)；

2. DF.drop('column_name',axis=1, inplace=True)

3. DF.drop([DF.columns[[0,1, 3]]], axis=1,inplace=True)  # Note: zero indexed

'''
```




    "\n\n方法一：直接del DF['column-name']\xa0\n\n方法二：采用drop方法，有下面三种等价的表达式：\n\n1. DF= DF.drop('column_name', 1)；\n\n2. DF.drop('column_name',axis=1, inplace=True)\n\n3. DF.drop([DF.columns[[0,1, 3]]], axis=1,inplace=True)\xa0 # Note: zero indexed\n\n"



#### 1.4.6 任务六： 将['PassengerId','Name','Age','Ticket']这几个列元素隐藏，只观察其他几个列元素


```python
df.drop(['乘客id','姓名','年龄','船票信息'],axis=1).head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>性别</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



【思考】对比任务五和任务六，是不是使用了不一样的方法(函数)，如果使用一样的函数如何完成上面的不同的要求呢？

【思考回答】

如果想要完全的删除你的数据结构，使用inplace=True，因为使用inplace就将原数据覆盖了，所以这里没有用

### 1.5 筛选的逻辑

表格数据中，最重要的一个功能就是要具有可筛选的能力，选出我所需要的信息，丢弃无用的信息。

下面我们还是用实战来学习pandas这个功能。

#### 1.5.1 任务一： 我们以"Age"为筛选条件，显示年龄在10岁以下的乘客信息。


```python
df[df["年龄"]<10].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>乘客id</th>
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.075</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.700</td>
      <td>G6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Master. Eugene</td>
      <td>male</td>
      <td>2.0</td>
      <td>4</td>
      <td>1</td>
      <td>382652</td>
      <td>29.125</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.5.2 任务二： 以"Age"为条件，将年龄在10岁以上和50岁以下的乘客信息显示出来，并将这个数据命名为midage


```python
midage = df[(df["年龄"]>10)&(df["年龄"]<50)]
midage.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>乘客id</th>
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



【提示】了解pandas的条件筛选方式以及如何使用交集和并集操作

#### 1.5.3 任务三：将midage的数据中第100行的"Pclass"和"Sex"的数据显示出来


```python
# 数据清洗时，会将带空值的行删除，此时DataFrame或Series类型的数据不再是连续的索引，可以使用reset_index()重置索引, 不想保留原来的index，
# 使用参数 drop=True，默认 False

midage = midage.reset_index(drop=True)
midage.loc[[100],['仓位等级','性别']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>仓位等级</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>2</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>



【思考】这个reset_index()函数的作用是什么？如果不用这个函数，下面的任务会出现什么情况？

#### 1.5.4 任务四：将midage的数据中第100，105，108行的"Pclass"，"Name"和"Sex"的数据显示出来


```python
midage.loc[[100,105,108],['仓位等级','姓名','性别']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>2</td>
      <td>Byles, Rev. Thomas Roussel Davids</td>
      <td>male</td>
    </tr>
    <tr>
      <th>105</th>
      <td>3</td>
      <td>Cribb, Mr. John Hatfield</td>
      <td>male</td>
    </tr>
    <tr>
      <th>108</th>
      <td>3</td>
      <td>Calic, Mr. Jovo</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>



【提示】使用pandas提出的简单方式，你可以看看loc方法

对比整体的数据位置，你有发现什么问题吗？那么如何解决？

#### 1.5.5 任务五：使用iloc方法将midage的数据中第100，105，108行的"Pclass"，"Name"和"Sex"的数据显示出来


```python
midage.iloc[[100,105,108],[2,3,4]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>2</td>
      <td>Byles, Rev. Thomas Roussel Davids</td>
      <td>male</td>
    </tr>
    <tr>
      <th>105</th>
      <td>3</td>
      <td>Cribb, Mr. John Hatfield</td>
      <td>male</td>
    </tr>
    <tr>
      <th>108</th>
      <td>3</td>
      <td>Calic, Mr. Jovo</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
