这是《基于多智能体深度强化学习的大规模车辆换道控制》的代码实现。

环境的设计基于highway-env (<https://github.com/Farama-Foundation/HighwayEnv>)
MaHighwayEnv是该环境的一个修改版本，旨在提供更多的受控车辆并额外设置起点和终点，当车辆到达终点后会被清除，每秒都会按照预设分布在起点生成受控车辆或其他车辆。

论文提供的方法不依赖与环境，代码中提供了算法的完整实现。受限于其他项目，环境代码不能公开，需要有需要可以交流实现细节，相同功能的环境代码应该能得到与论文相似的结果。下面是我用的环境其他安装包情况

pip install  gym==0.15.4 gymnasium==0.28.1 imageio==2.31.2 importlib-metadata==6.7.0 jax-jumpy==1.0.0 
matplotlib==3.5.3 numpy==1.21.6 pandas==1.1.5 pygame==2.5.0 scipy==1.7.3 
torch==1.9.1+cu102
I