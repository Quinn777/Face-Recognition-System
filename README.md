# Face-Recognition-System
使用MTCNN+MobileFaceNet构建了一个人类识别系统，在LFW与自定义数据集上达到了较高的识别精度，并尝试使用importance-score进行模型压缩
run：
```
python test_video.py
```
result:
| prune rate | LFW   | ourself |
| ---------- | ----- | ------- |
| 1.0        | 98.76 | 99.09   |
| 0.8        | 99.30 | 72.03   |
| 0.5        | 96.46 | <=10    |

