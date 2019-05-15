import cPickle as pickle
from sklearn import metrics
# from scipy.io import loadmat
# from sklearn.datasets import fetch_lfw_people
# import matplotlib.pyplot as plt
# faces = fetch_lfw_people()

# label = loadmat("../data/pairlist_lfw.mat")
# pairlist_lfw = label['pairlist_lfw'][0][0][0]
# print(type(label),type(pairlist_lfw),pairlist_lfw.shape)

# print(pairlist_lfw)

# a = pairlist_lfw[0]
# print(a)
# face = faces.images[a[0]]
# plt.imshow(face,cmap=plt.cm.gray)
# plt.show()
# face = faces.images[a[1]]
# plt.imshow(face,cmap=plt.cm.gray)
# plt.show()

with open("../result/result.pkl", "rb") as f:
    result = pickle.load(f)


dist = result['distance']
y    = result['label']

print y
print "test size: ", y.shape
print "negative size: ", y[y==0].shape
print "postive size: ",  y[y==1].shape

draw_list = []
pre = dist >= -16.9
y = (y==1)
report = metrics.classification_report(y_true=y, y_pred=pre)


print report
print metrics.accuracy_score(y,pre)
print metrics.recall_score(y,pre)
print metrics.precision_score(y,pre)
print metrics.f1_score(y,pre)

# for i,j in enumerate(zip(pre,y)):
#     p = j[0]
#     yy = j[1]
#     if p!=yy:
#         print(i,p,yy)

