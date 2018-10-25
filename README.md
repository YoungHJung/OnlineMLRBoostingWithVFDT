# Online Multi-label Ranking Boosting
A Python implementation of online multi-label ranking boosting using VFDT as weak learners. 

The algorithms are described and theoretically analayzed in the following work. 
```
Young Hun Jung and Ambuj Tewari. 
Online Boosting Algorithms for Multi-label Ranking.
In AISTATS 2018.
```

If you use this code in your paper, please cite the above work. Although it is based on this we cannot guarantee that the algorithm will work exactly, or even produce the same output, as any of these implementations.

For our weak learners, we used the VFDT proposed and implemented by the following two works. 

```
Pedro Domingos and Geoff Hulten. 2000. 
Mining high-speed data streams.
In Proceedings of the sixth ACM SIGKDD international conference on
  Knowledge discovery and data mining (KDD '00). 
ACM, New York, NY, USA, 71-80.
```

```
Vitor da Silva and Ana Trindade Winck. 2017.
Video popularity prediction in data streams based on context-independent features. 
In Proceedings of the Symposium on Applied Computing (SAC '17). 
ACM, New York, NY, USA, 95-100. 
DOI: https://doi.org/10.1145/3019612.3019638
```

The folders "core" and "ht", and the file "hoeffdingtree.py" are copied from the above works, and only minor changes are made to be compatible in python 2.7 (The original code was written in python 3.x). 

The folder "data" contains five data sets from MULAN. (see http://mulan.sourceforge.net/datasets.html)

```
Grigorios Tsoumakas, Eleftherios Spyromitros-Xioufis, Jozef Vilcek, and Ioannis Vlahavas.
Mulan: A java library for multi-label learning.
Journal of Machine Learning Research, 12:2411–2414, 2011.
```