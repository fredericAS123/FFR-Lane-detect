# FFR-Lane-detect
 A host computer algorithm that ensures quadrupedal robots do not yaw by detecting the track and making decisions.
## OP 文件夹：
为本算法初始版本，因工程时限，作者彼时能力有限，代码并不简洁，很多地方有待优化，但效果已被验证，准确性较好，可以满足正常需求。
## OO 文件夹:
为本算法的面向对象版本，方便移植，准确性有待考证。
## learing basis 文件夹：
为学习过程中所学习和扩展的代码集合，同时收录了智能车竞赛的优秀代码与资料。
## verification 文件夹：
展示了算法的良好效果。
通过Unity仿真实现的赛道，具有一定阴影和多余边线等挑战因素：
![unity](/verification/unity.png)

可以看到，分割十分干净

![edge seg](/verification/edges.png)

算法最大程度上防止了多余边线对判断的影响

![cropped](/verification/cropped.png)

判断结果准确

![result](/verification/result.png)

### 具体介绍见每个文件的提交消息
## HTTP download 到本地：
```
  git clone https://github.com/fredericAS123/FFR-Lane-detect.git
```
喜欢点个star哦
