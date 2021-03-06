import numpy as np
data = np.load('score_tr.npy')
score_title = np.zeros(4 * 7 + 3).tolist()

for i in range(0,4):
  if i == 0:
    prefix = 'feature pixel'
  elif i == 1:
    prefix = 'feature conv'
  elif i == 2:
    prefix = 'feature logit'
  elif i == 3:
    prefix = 'feature smax'
  score_title[i * 7] = "wasserstein {}".format(prefix)
  score_title[i * 7 + 1] = "mmd {}".format(prefix)
  score_title[(i * 7 + 2):(i * 7 + 7)] = "knn acc {}".format(prefix), "knn acc real {}".format(prefix), "knn acc fake {}".format(prefix), "knn acc precision {}".format(prefix), "knn acc recall {}".format(prefix)
score_title[28] ="inception score"
score_title[29] ="mode score"
score_title[30] ="fid"

for i, d in enumerate(data):
    print('Epochs: {}'.format(i))
    for j, metrics in enumerate(d):
        print('Metrix {}: {}'.format(score_title[j], metrics) )
    print('='*50)