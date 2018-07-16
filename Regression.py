import numpy as np
import itertools
import matplotlib.pyplot as plt

#-----------------------------------------------------
# 回帰分析
#-----------------------------------------------------

# ステップ
step_a = 0.005
step_b = 0.005

# 正解データ
a = 0.3
b = 0.1
x = np.linspace(0,100,101)
rand = np.random.normal(0,1,101)
y = a*x+b+rand

# 回帰データ
a_reg = 0
b_reg = 0
y_reg = a_reg*x + b_reg

# 計算
for i in range(10000):
    y_reg = a_reg*x + b_reg  # 再計算
    e = np.sum((y-y_reg)**2) # 目的関数　最小二乗誤差

    if np.abs(e) < 150:
        break
    e_a = np.sum((y-((a_reg+step_a)*x+b_reg))**2)-np.sum((y-(a_reg*x+b_reg))**2)
    e_diffb = np.sum((y-(a_reg*x+(b_reg+step_b)))**2)-np.sum((y-(a_reg*x+b_reg))**2)

    if(e_a > 0):
        a_reg -= step_a
    elif(e_a < 0):
        a_reg += step_a

    if(e_diffb > 0):
        b_reg -= step_b
    elif(e_diffb < 0):
        b_reg += step_b

# 決定係数
y = a*x + b+rand
y_reg = a_reg*x + b_reg
R = 1 - np.sum((y - y_reg)**2)/np.sum((y-np.mean(y))**2)

# 推定値
print(e)
print(i)
print(a_reg)
print(b_reg)
print(R)

# 結果表示
plt.plot(x,y)
plt.plot(x,y_reg)
plt.show()
#-----------------------------------------------------
# 重回帰分析
#-----------------------------------------------------

# 回帰モデル用変数
num_expvar = 5
const = 0
weight = np.ones(num_expvar)

# 最急降下法：ステップ
step_const = 0.001
step_weight = np.ones(num_expvar)*0.001
step_weight_diag = np.diag(step_weight)

# 正解データ
num_sumple = 100
contrib = np.random.rand(num_expvar)
contrib /= np.sum(contrib)
expvar = np.random.rand(num_expvar,num_sumple)
expconst = np.random.rand()
z = np.dot(contrib,expvar)+expconst

# 最急降下法：重み(一時保存用)
const_tmp = const
weight_tmp = weight

# 計算
for i in range(10000):

    # 目的関数：最小二乗
    z_reg = np.dot(weight,expvar)+const
    sqerror = np.sum((z_reg - z)**2)
    if(sqerror < 10**(-2)):
        print(sqerror)
        print(i)
        break

    # 最急降下法：定数項
    z_dif = np.dot(weight,expvar) + (const + step_const)
    sqerror_dif = np.sum((z_dif - z)**2)
    if(sqerror_dif - sqerror > 0):
        const_tmp -= step_const
    elif(sqerror_dif - sqerror < 0):
        const_tmp += step_const

    # 最急降下法：重み
    for w, item in enumerate(weight):
        z_dif = np.dot(weight+step_weight_diag[w],expvar) + const
        sqerror_dif = np.sum((z_dif - z)**2)
        if(sqerror_dif - sqerror > 0):
            weight_tmp[w] -= step_weight[w]
        elif(sqerror_dif - sqerror < 0):
            weight_tmp[w] += step_weight[w]

    # 最急降下法：定数項・重み更新
    const = const_tmp
    weight = weight_tmp

# 決定係数
z = np.dot(contrib,expvar)+expconst
z_reg = np.dot(weight,expvar)+const
flexibility = num_sumple - num_expvar - 1
R_numer = np.sum((z-z_reg)**2)/flexibility
R_denom = np.sum((z-np.mean(z))**2)/(num_sumple - 1)
R = 1 - R_numer/R_denom

# 多重共線性の確認(説明変数の相互相関)
combos = list(itertools.combinations(np.arange(num_expvar),2))
corr_expvar = []
for combo in combos:
    corr_expvar.append(np.correlate(expvar[combo[0]]-np.mean(expvar[combo[0]]),
                                    expvar[combo[1]]-np.mean(expvar[combo[1]]), "full")[0])
dict_corr_expvar = {"combo": combos,"corr_expvar":corr_expvar}

# 推定値
print(contrib)
print(weight)
print(R)
print(dict_corr_expvar)

# 結果表示
plt.bar(range(num_expvar),contrib,1)
plt.show()

#-----------------------------------------------------
# ロジスティック回帰
#-----------------------------------------------------

# 回帰モデル用変数
# z = weight[0]*expvar[0]+weight[1]*expvar[1]*...+const
num_expvar = 5
const = 1
weight = np.ones(num_expvar)*0

# 最急降下法：ステップ
step_const = 0.1
step_weight = np.ones(num_expvar)*0.001
step_weight_diag = np.diag(step_weight)

# 正解データ
threshold = 0.5
avoid_log0_1 = 1 + 10**(-10)                        # nearly equal 1 / to avoid log(0)
expvar = np.random.rand(num_expvar,100)             # 説明変数
contrib = np.random.rand(num_expvar)
contrib /= np.sum(contrib)
z = np.dot(contrib,expvar)
label = np.array(z > threshold, dtype ="float32")

plt.scatter(z,label)
plt.show()

# 最急降下法：重み(一時保存用)
const_tmp = const
weight_tmp = weight

for i in range(100000):

    # 最急降下法：対数尤度関数
    z_reg = const + np.dot(weight,expvar)
    sigmoid = 1 / (1 + np.exp(-z_reg))
    likeli = np.sum(label*np.log(sigmoid)+(1-label)*np.log(avoid_log0_1-sigmoid))
    if(likeli > -1):
        print(likeli)
        print(i)
        break

    # 最急降下法：定数項
    z_dif = (const + step_const) + np.dot(weight,expvar)
    sigmoid_dif = 1 / (1 + np.exp(-z_dif))
    likeli_dif = np.sum(label*np.log(sigmoid_dif)+(1-label)*np.log(avoid_log0_1-sigmoid_dif))
    if(likeli_dif - likeli > 0):
        const_tmp += step_const
    elif(likeli_dif - likeli < 0):
        const_tmp -= step_const

    # 最急降下法：重み
    for w, item in enumerate(weight):
        z_dif = const + np.dot(weight+step_weight_diag[w],expvar)
        sigmoid_dif = 1 / (1 + np.exp(-z_dif))
        likeli_dif = np.sum(label*np.log(sigmoid_dif)+(1-label)*np.log(avoid_log0_1-sigmoid_dif))
        if(likeli_dif - likeli > 0):
            weight_tmp[w] += step_weight[w]
        elif(likeli_dif - likeli < 0):
            weight_tmp[w] -= step_weight[w]

    # 最急降下法：定数項・重み更新
    const = const_tmp
    weight = weight_tmp

# 多重共線性の確認(説明変数の相互相関)
combos = list(itertools.combinations(np.arange(num_expvar),2))
corr_expvar = []
for combo in combos:
    corr_expvar.append(np.correlate(expvar[combo[0]]-np.mean(expvar[combo[0]]),
                                    expvar[combo[1]]-np.mean(expvar[combo[1]]), "full")[0])
dict_corr_expvar = {"combo": combos,"corr_expvar":corr_expvar}

# 推定値
print(contrib)
print(const)
print(weight/np.sum(weight))
print(dict_corr_expvar)

# 結果表示
z_reg = const + np.dot(weight,expvar)
sigmoid = 1 / (1 + np.exp(-z_reg))

plt.scatter(z,label)
plt.scatter(z,sigmoid)
plt.show()
plt.bar(np.arange(num_expvar),contrib/np.sum(contrib), color="r", label="contrib")
plt.bar(np.arange(num_expvar)+0.4,weight/np.sum(weight), color="b", label="weight")
plt.show()
