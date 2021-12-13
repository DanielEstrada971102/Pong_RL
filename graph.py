import matplotlib.pyplot as plt


entropy_loss=[]
explained_variance =[]
learning_rate =[]
policy_loss =[]
value_loss =[]

with open("td_RL_MF_env_A2C.txt", 'r') as f:
    for row in f:
        if row[0:2] == '--':
            continue
        if row[1:24]=='    entropy_loss       ':
            entropy_loss.append(float(row[26:-2]))
        if row[1:24]=='    explained_variance ':
            explained_variance.append(float(row[26:-2]))
        if row[1:24]=='    learning_rate      ':
            learning_rate.append(float(row[26:-2]))
        if row[1:24]=='    policy_loss        ':
            policy_loss.append(float(row[26:-2]))
        if row[1:24]=='    value_loss         ':
            value_loss.append(float(row[26:-2]))
            
#print(entropy_loss,explained_variance, learning_rate, policy_loss, value_loss)

plt.plot(range(500, 80001, 500),entropy_loss)
plt.show()

