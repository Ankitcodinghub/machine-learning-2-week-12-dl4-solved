# machine-learning-2-week-12-dl4-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning 2 Week 12-DL4 Solved](https://www.ankitcodinghub.com/product/machine-learning-2-week-12-dl4-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98852&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning 2 Week 12-DL4 Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column"></div>
<div class="column">
&nbsp;

Exercise Sheet 12

</div>
</div>
<div class="layoutArea">
<div class="column">
Consider a dataset x1, . . . , xN ∈ Rd, and a simple linear feature map φ(x) = w⊤x + b with trainable parameters w and b. For this simple scenario, we can formulate the deep SVDD problem as:

w,b N i=1

where we have hardcoded the center parameter of deep SVDD to 1. We then classify new points x to be

anomalous if ∥w⊤x + b − 1∥2 &gt; τ.

<ol>
<li>(a) &nbsp;Give a choice of parameters (w, b) that minimizes the objective above for any dataset (x1, . . . , xN ).</li>
<li>(b) &nbsp;We now consider a regularizer for our feature map φ which simply consists of forcing the bias term to b = 0. Show that under this regularizer, the solution of deep SVDD is given by:
w = Σ − 1 x ̄

where x ̄ and Σ are the empirical mean and uncentered covariance.

Exercise 2: Restricted Boltzmann Machine (30 P)

The restricted Boltzmann machine is a system of binary variables comprising inputs x ∈ {0, 1}d and hidden units h ∈ {0, 1}K . It associates to each configuration of these binary variables the energy:

E(x, h) = −x⊤W h − b⊤h and the probability associated to each configuration is then given as:

p(x, h) = Z1 exp(−E(x, h))

where Z is a normalization constant that makes probabilities sum to one. Let sigm(t) = exp(t)/(1 + exp(t))

be the sigmoid function.
</li>
</ol>
<ol>
<li>(a) &nbsp;Show that p(hk = 1 | x) = sigm􏰍x⊤W:,k + bk􏰎.</li>
<li>(b) &nbsp;Show that p(xj = 1|h) = sigm􏰍W⊤ h􏰎. j,:</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
(c) Show that

where

</div>
<div class="column">
p(x) = Z1 exp(−F (x))

K F(x)=−􏰄log􏰍1+exp􏰍x⊤W:,k +bk􏰎􏰎

</div>
</div>
<div class="layoutArea">
<div class="column">
k=1

is the free energy and where Z is again a normalization constant.

Exercise 3: Programming (50 P)

Download the programming files on ISIS and follow the instructions.

</div>
</div>
<div class="layoutArea">
<div class="column">
1 􏰄N

min ∥w⊤xi + b − 1∥2

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="section">
<div class="layoutArea">
<div class="column">
Exercise sheet 12 (programming) [SoSe 2021] Machine Learning 2

</div>
</div>
<div class="layoutArea">
<div class="column">
KDE and RBM for Anomaly Detection

In this programming exercise, we compare in the context of anomaly detection two energy-based models: kernel density estimation (KDE) and the restricted Boltzmann machine (RBM).

</div>
</div>
<div class="layoutArea">
<div class="column">
In [1]:

</div>
<div class="column">
import utils

import numpy

import scipy,scipy.special,scipy.spatial import sklearn,sklearn.metrics %matplotlib inline

import matplotlib

from matplotlib import pyplot as plt

</div>
</div>
<div class="layoutArea">
<div class="column">
We consider the MNIST dataset and define the class “0” to be normal (inlier) and the remain classes (1-9) to be anomalous (outlier). We consider that we have a training set Xr composed of 100 normal data points. The variables

Xi and Xo denote normal and anomalous test data. In [2]: Xr,Xi,Xo = utils.getdata()

The 100 training points are visualized below:

In [3]: plt.figure(figsize=(16,4)) plt.imshow(Xr.reshape(5,20,28,28).transpose(0,2,1,3).reshape(140,560)) plt.show()

</div>
</div>
<div class="layoutArea">
<div class="column">
Kernel Density Estimation (15 P)

We first consider kernel density estimation which is a shallow model for anomaly detection. The code below implement kernel density estimation.

Task:

Implement the function energy that returns the energy of the points X given as input as computed by the KDE energy function (cf. slide Kernel Density Estimation as an EBM).

</div>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
In [4]: class AnomalyModel:

</div>
</div>
<div class="layoutArea">
<div class="column">
def auroc(self):

Ei = self.energy(Xi)

Eo = self.energy(Xo)

return sklearn.metrics.roc_auc_score(

<pre>            numpy.concatenate([Ei*0+0,Eo*0+1]),
</pre>
<pre>            numpy.concatenate([Ei,Eo])
        )
</pre>
class KDE(AnomalyModel):

def __init__(self,gamma):

<pre>        self.gamma = gamma
</pre>
def fit(self,X): self.X = X

def energy(self,X):

# ———————————————— # TODO: Replace by your code

# ———————————————— import solution

<pre>        E = solution.kde_energy(self,X)
</pre>
<pre>        # ------------------------------------------------
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
return E

The following code applies KDE with different scale parameters gamma

anomaly detection model measured in terms of area under the ROC.

In [5]: for gamma in numpy.logspace(-2,0,10):

</div>
<div class="column">
and returns the performance of the resulting

</div>
</div>
<div class="layoutArea">
<div class="column">
kde = KDE(gamma)

kde.fit(Xr)

print(‘gamma = %5.3f AUROC = %5.3f’%(gamma,kde.auroc()))

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>              gamma = 0.010  AUROC = 0.957
              gamma = 0.017  AUROC = 0.962
              gamma = 0.028  AUROC = 0.969
              gamma = 0.046  AUROC = 0.976
              gamma = 0.077  AUROC = 0.981
              gamma = 0.129  AUROC = 0.983
              gamma = 0.215  AUROC = 0.983
              gamma = 0.359  AUROC = 0.982
              gamma = 0.599  AUROC = 0.982
              gamma = 1.000  AUROC = 0.981
</pre>
We observe that the best performance is obtained for some intermediate value of the parameter

</div>
<div class="column">
gamma .

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="section">
<div class="layoutArea">
<div class="column">
Restricted Boltzmann Machine (35 P)

We now consider a restricted Boltzmann machine composed of 100 binary hidden units (h ∈ {0, 1}100 ). The joint energy function of our RBM is given by:

E(x, h) = −x⊤ a − x⊤ W h − h⊤ b

The model can be marginalized over its hidden units and the energy function that depends only on the input x is then

given as:

100 E(x)=−x⊤a−∑log(1+exp(x⊤W:,k +bk))

k=1 The RBM training algorithm is already implemented for you.

Tasks:

Implement the energy function E(x)

Augment the function fit with code that prints the AUROC every 100 iterations.

</div>
</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
In [6]:

</div>
<div class="column">
def sigm(t): return numpy.tanh(0.5*t)*0.5+0.5

def realize(t): return 1.0*(t&gt;numpy.random.uniform(0,1,t.shape))

class RBM(AnomalyModel):

def __init__(self,X,h): self.mb = X.shape[0] self.d = X.shape[1] self.h = h

self.lr = 0.1

<pre>        # Model parameters
</pre>
<pre>        self.A = numpy.zeros([self.d])
</pre>
<pre>        self.W = numpy.random.normal(0,self.d**-.25 * self.h**-.25,[self.
d,self.h])
</pre>
self.B = numpy.zeros([self.h]) def fit(self,X,verbose=False):

Xm = numpy.zeros([self.mb,self.d]) for i in numpy.arange(1001):

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>0.01*self.W)
</pre>
</div>
<div class="column">
<pre># Gibbs sampling (PCD)
</pre>
<pre>Xd = X*1.0
Zd = realize(sigm(Xd.dot(self.W)+self.B))
Zm = realize(sigm(Xm.dot(self.W)+self.B))
Xm = realize(sigm(Zm.dot(self.W.T)+self.A))
</pre>
<pre># Update parameters
</pre>
<pre>self.W += self.lr*((Xd.T.dot(Zd) - Xm.T.dot(Zm)) / self.mb -
</pre>
<pre>self.B += self.lr*(Zd.mean(axis=0)-Zm.mean(axis=0))
self.A += self.lr*(Xd.mean(axis=0)-Xm.mean(axis=0))
</pre>
if verbose:

# ———————————————— # TODO: Replace by your code

# ———————————————— import solution

solution.track_auroc(self,i)

# ————————————————

</div>
</div>
<div class="layoutArea">
<div class="column">
def energy(self,X):

# ———————————————— # TODO: Replace by your code

# ———————————————— import solution

<pre>    E = solution.rbm_energy(self,X)
</pre>
<pre>    # ------------------------------------------------
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
return E

We now train our RBM on the same data as the KDE model for approximately 1000 iterations.

</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
In [7]:

</div>
<div class="column">
rbm = RBM(Xr,100) rbm.fit(Xr,verbose=True)

it= 0 AUROC = 0.962 it = 100 AUROC = 0.943 it = 200 AUROC = 0.985 it = 300 AUROC = 0.987 it = 400 AUROC = 0.988 it = 500 AUROC = 0.986 it = 600 AUROC = 0.987 it = 700 AUROC = 0.987 it = 800 AUROC = 0.989 it = 900 AUROC = 0.986 it = 1000 AUROC = 0.990

</div>
</div>
<div class="layoutArea">
<div class="column">
We observe that the RBM reaches superior levels of AUROC performance compared to the simple KDE model. An advantage of the RBM model is that it learns a set of parameters that represent variations at multiple scales and with specific orientations in input space. We would like to visualize these parameters:

Task:

Render as a mosaic the weight parameters ( W ) of the model. Each tile of the mosaic should correspond to the receptive field connecting the input image to a particular hidden unit.

</div>
</div>
<div class="layoutArea">
<div class="column">
In [8]:

</div>
<div class="column">
# ———————————————— # TODO: Replace by your code

# ———————————————— import solution

<pre>solution.plot_weights(rbm)
</pre>
<pre># ------------------------------------------------
</pre>
</div>
</div>
</div>
<div class="page" title="Page 7"></div>
<div class="page" title="Page 8"></div>
