好家伙 已经4、50天没有写公众号了，绝不是因为最近瓜太多了，而是单纯的懒癌犯了。寒假已经开始了，小伙伴们有什么激动人心的计划吗？（随便寒暄几句，水个开头）

不多bb，今天的内容是之前文章的续集，**文章名称**最后开了个坑，说有空的时候要补上遗传算法在路径求解问题中的应用，这次就给大家整一个。

## 问题描述

简单来说，本次的例子是一个多目标无向图求最短路径的问题（该问题的原型来自于我同门的论文，危货的路径优化）。为简单起见，使用下图中的网络结构，图片是从知乎上弄的，来源于这篇文章https://zhuanlan.zhihu.com/p/81749290，知乎原文是使用geatpy求解下图的最短路，本文代码也仅是在其基础上稍作修改。

![0fe5f6aa030c423da265ad7744f01f4](https://i.loli.net/2021/01/22/V2j5MpSiEAaCufL.jpg)

具体的目标函数有3个，均为最小化。第一个目标是最小化起终点间的路径时间阻抗，第二个目标是最小化风险阻抗，第三个目标是最小化路径上最大的路段风险值与最小的路段风险值之间的差值。当然，各位大佬们可以自己建立不同的模型，此处仅是举个例子。

本文就使用上图中的权重值当作此次的路段时间阻抗，风险阻抗值是在Excel里面randbetween随机生成的1到10之间的数值（有需要的可以自提，代码也在一起，这个地址https://github.com/Bardbo/geatpy-Multi-objective-Optimization-for-Shortest-Path）。知乎那篇文章中是单目标的有向图，这里将前面的图当成是无向图即可。

## 相关知识介绍

主要是三个知识点吧，我不怎么会遗传算法，所以只能随便讲讲哈。这三个个人觉得比较重要的点分别如下：

+ 多目标求解的相关理论，主要是 Pareto最优解（非支配集）
+ 路径优化问题如何编码，主要是基于优先级的编码
+ 无向图中如何避免回路等（当然有向图也可能存在回路）

### 多目标优化

geatpy的官方文档讲的很详细，真的用遗传算法很推荐这个python库，国产良心。下面内容节选自官方文档，建议大家跳过，直接去官网看，百度搜索geatpy-->文档-->进化算法介绍-->第七章：多目标优化，具体地址是http://geatpy.com/index.php/2019/07/28/%E7%AC%AC%E4%B8%83%E7%AB%A0%EF%BC%9A%E5%A4%9A%E7%9B%AE%E6%A0%87%E4%BC%98%E5%8C%96/。

多个目标之间可能会拥有不同的单位，也可能在优化某个目标时损害其他目标。但这并不意味着多目标优化问题可能没有最优解，事实上是可以有的，为了求出比较合理的解，这里引入多目标优化问题的合理解集——Pareto最优解(pareto optimal solution)。

设X1, X2均是解空间ω中的解，若对于所有的目标，均有X1比X2好，那么就称X1强支配X2。若对于所有的目标，均有X1不差于X2，且至少存在一个目标使得X1比X2好，那么就称X1弱支配X2。

若X1对于解空间中的其他解而言X1都不是被支配的，那么此时X1就是一个帕累托最优解。

### 路径编码

前面的知乎文章提到了两种编码方式，第一种就是直接使用01编码，用到哪条边哪条边的编码就是1，没有用到就是0，对于节点也适用。这种编码方式弊端就是染色体的编码往往无法表达合理的路径，因为没有考虑到节点间的邻接性。

第二种就是基于优先级的编码方式了。这是一种间接编码方式，染色体不能直接表示路径。具体的也可以参考知乎原文的介绍，下面简单举例说明（实则也是抄的知乎原文）。

假设在上述问题中有这么一条染色体，如下：

![v2-798995c676566d804190a67211f2dc11_720w](D:\Users\Super\Desktop\v2-798995c676566d804190a67211f2dc11_720w.png)

求解的是1到10的一条路径，那么首先来看1的邻居节点有哪些，是2和3，然后因为3的染色体数值大于2的（优先级比较高），所以从1出发解码后会到3。其余的同样看邻居节点和优先级，依次选择优先级高的邻居节点即可。

这里需要注意一点，知乎的是有向图，本文是无向图，因此一些点的邻居节点会多一些。

### 如何避免回路

知乎原文中提了一嘴——处理方式有很多，例如在解码过程中对已经访问过的结点的有限度进行惩罚等等。但是该作者没有出后续，然后加上我也懒得看书看论文就自己随便想了一种处理方式，就是在每次解码的过程中去掉原本已经访问过的节点，这个去掉是指在邻居节点里去掉（邻居节点里去掉就相当于去掉了可以前往这个节点的路），这种方法就使得路径不能重复经过一个点两次了，但是也无法保证该路径能走到终点了，如果路网中存在死胡同的话就只能走到哪算哪了。这种方法如果加一个可以经过几次的限制条件然后到达了该限制再去除节点应该就和原作者说的有限度惩罚相类似了吧，好像没有这样做的必要，我这里就直接不准它经过同个点两次了。

具体的操作实现写在代码里了，与上述思路是一致的。

## 代码部分

代码主要有3个部分，第一个部分是数据读取等，第二个部分是问题类，第三个部分调用模板求解。

第一部分中结合networkx库获取了节点的邻居（这步也可以自己写遍历，从原数据中获取），然后同时也使用字典结构存储了各条边的时间阻抗和风险阻抗，便于后面目标函数的计算。具体代码如下：

```python
import numpy as np
import geatpy as ea
import pandas as pd
import networkx as nx
import copy

# geatpy版本为2.6.0
# ea.__version__

# 读取数据
data = pd.read_excel('data.xlsx')
# 构建图网络，端点1和2表示图中边的端点号，一条边两个端点
g = nx.Graph()
g.add_edges_from([(i,j) for i,j in zip(data['端点1'], data['端点2'])])
# 获取各点的邻居节点，存储在字典中
neighbors = {i:[j for j in g[i]] for i in g.nodes}
# 获取各条边的风险阻抗值，存储在字典中，由于是无向的，即等价于双向，两个方向都记录了一次
# 如果是有向图就不用反过来记录一遍了，同时前面的邻居节点中也应该对应的去除一些邻居
risk_weights = {f'({i}, {j})':k for i,j,k in zip(data['端点1'], data['端点2'], data['风险值'])}
risk_weights.update({f'({j}, {i})':k for i,j,k in zip(data['端点1'], data['端点2'], data['风险值'])})
# 同理获取各条边的时间阻抗值
time_weights = {f'({i}, {j})':k for i,j,k in zip(data['端点1'], data['端点2'], data['阻抗值'])}
time_weights.update({f'({j}, {i})':k for i,j,k in zip(data['端点1'], data['端点2'], data['阻抗值'])})
```

接下来就是定义问题类，初始化部分没啥好讲的，这里主要看的是解码部分和目标函数部分，代码参考了知乎文章。

解码的decode函数是将优先级染色体解码为路径，采用while循环的方式从起点开始解码，如果走入了死胡同或者走到了终点就停止解码，在每次的解码过程中将已经走过了的点从邻居节点字典中去除。

目标函数计算的aimFunc函数是将传入的种群遍历解码并计算目标函数，这里有三个目标函数，因此在计算后将其拼接起来合成ObjV矩阵，**值得注意的是没能走到终点的线路应该加以惩罚**，这种惩罚在geatpy中有两种处理方式，一种是罚函数，即对该种解的目标函数值进行惩罚，另外一种是通过CV矩阵表示可行性，值大于0表示不可行，越大越不可行，这里使用后一种方式，将对应不可行解的cv值设置为10000。使用第一种方式也可以，但是geatpy官方一般推荐使用CV矩阵的形式。

代码如下：

```python
class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self):
        name = 'Shortest_Path' # 初始化name（函数名称，可以随意设置）
        M = 3 # 初始化M（目标维数）
        maxormins = [1] * M # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 10 # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [1] * Dim # 决策变量下界
        ub = [10] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界
        ubin = [1] * Dim # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 设置每一个结点下一步可达的结点（结点从1开始数，因此列表nodes的第0号元素设为空列表表示无意义）
        self.neighbors = neighbors
        # 设置有向图中各条边的权重
        self.risk_weights = risk_weights
        self.time_weights = time_weights
    
    def decode(self, priority): # 将优先级编码的染色体解码得到一条从节点1到节点10的可行路径
        edges = [] # 存储边
        path = [1] # 结点1是路径起点
        new_neighbors = copy.deepcopy(self.neighbors)
        while not path[-1] == 10: # 开始从起点走到终点
            currentNode = path[-1] # 得到当前所在的结点编号
            nextNodes = new_neighbors[currentNode] # 获取下一步可达的结点组成的列表
            if nextNodes:
                chooseNode = nextNodes[np.argmax(priority[np.array(nextNodes) - 1])] # 从NextNodes中选择优先级更高的结点作为下一步要访问的结点，因为结点从1数起，而下标从0数起，因此要减去1
                path.append(chooseNode)
                edges.append((currentNode, chooseNode))
                for k,v in new_neighbors.items():
                    # 此处移除已经走过的点，注意到未移除遍历过的k，没有必要
                    if chooseNode in v:
                        v.remove(chooseNode)
                        new_neighbors[k] = v
            else:
                break
        return path, edges

    def aimFunc(self, pop): # 目标函数
        f1_objv, f2_objv, f3_objv = np.zeros((pop.sizes, 1)), np.zeros((pop.sizes, 1)), np.zeros((pop.sizes, 1)) # 初始化ObjV
        cv = np.zeros((pop.sizes, 1))
        for i in range(pop.sizes): # 遍历种群的每个个体，分别计算各个个体的目标函数值
            priority = pop.Phen[i, :]
            path, edges = self.decode(priority) # 将优先级编码的染色体解码得到访问路径及经过的边
            risk_value, time_value, link_risk = 0, 0, [10, 0]
            for edge in edges:
                key = str(edge) # 根据路径得到键值，以便根据键值找到路径对应的长度
#                 if not key in self.weights:
#                     raise RuntimeError("Error in aimFunc: The path is invalid. (当前路径是无效的。)", path)
                now_risk = self.risk_weights[key]
                risk_value += now_risk # 将该段风险值加入
                time_value += self.time_weights[key] # 将该段阻抗值加入
                if now_risk < link_risk[0]:
                    link_risk[0] = now_risk
                if now_risk > link_risk[1]:
                    link_risk[1] = now_risk
            f1_objv[i] = risk_value # 计算目标函数值
            f2_objv[i] = time_value
            f3_objv[i] = link_risk[1] - link_risk[0]
            if path[-1] != 10:
                cv[i] = 10000
        pop.CV = cv
        pop.ObjV = np.hstack([f1_objv, f2_objv, f3_objv])  # 把求得的目标函数值赋值给种群pop的ObjV
```

最后一部分是套模板，本次求解多目标函数使用的是moea_NSGA3_templet模板，也就是NSGA3算法。这个部分也没啥好说的，可以看看之前发的那篇推文，代码如下：

```python

if __name__ == '__main__':
    """===============================实例化问题对象============================"""
    problem = MyProblem()  # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'P'  # 编码方式
    NIND = 50  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.moea_NSGA3_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
    myAlgorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
    myAlgorithm.MAXGEN = 200  # 最大进化代数
    myAlgorithm.logTras = 0  # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = False  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化=========================
    调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
    详见Population.py中关于种群类的定义。
    """
    [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
    NDSet.save()  # 把非支配种群的信息保存到文件中
    """==================================输出结果=============================="""
    print('用时：%s 秒' % myAlgorithm.passTime)
    print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
```

这样就可以运行求解了，对于该问题一两秒就迭代完200代了，求解的非支配个体有45个，绘图如下：

![下载](D:\Users\Super\Desktop\下载.png)

之所以只有5个点，是因为多个不同的优先级染色体解码出来对应的路径可能是一样的，具体的线路可以自己解码出来，使用如下代码：

```python
best_ls = []
for i in range(45):
    best_journey, edges = problem.decode(NDSet.Phen[i])
    if best_journey not in best_ls:
        best_ls.append(best_journey)
```

这里的best_ls就是解码后的线路了，如下所示，五条

```python
[[1, 3, 2, 4, 8, 10],
 [1, 3, 6, 9, 10],
 [1, 3, 6, 9, 8, 10],
 [1, 2, 4, 7, 8, 10],
 [1, 3, 2, 4, 7, 8, 10]]
```

好啦，这就是多目标求解最短路的一个简单例子了，由于喵喵君水平有效，就只能到这样咯，进化算法领域虽然风评不太好（可能大家觉得水论文的多），但实际上也是很好用的算法，除了常用的优化问题外还可以应用于诸多领域，比如神经网络超参数调优、结构搜索、遗传编程等(当然也可以看成优化问题...)

我也没有深入学习就不多谈了~那有关于geatpy进化算法的介绍估计就只到这里了，小伙伴们 下期再见 寒假愉快！