import matplotlib.pyplot as plt
import numpy as np

L=2

for l in list(range(L)):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    name = 'sv_final_' + str(l) + '.dat'
    sv_file = open(name,'r')
    index_ = []
    qs_ = []
    svs_ = []
    sv_dict_ = {}
    index_dict_ = {}
    i=-1
    for line in sv_file:
        i=i+1
        ind,q,sv = line.strip().split("\t")
        index_.append(int(ind))
        
        if q == '()':
            qval = 0
        else:
            qval = int(q)
        qs_.append(qval)
        svs_.append(float(sv))

        if qval in sv_dict_:
            sv_dict_[qval].append(float(sv))
        else:
            sv_dict_[qval] = [float(sv)]
        if qval in index_dict_:
            index_dict_[qval].append(int(ind))
        else:
            index_dict_[qval] = [int(ind)]

    index = np.array(index_,dtype=int)
    qs = np.array(qs_,dtype=int)
    svs = np.array(svs_,dtype=float)
    
    sv_dict = {}
    for k, v in sv_dict_.items():
        sv_dict[k] = np.array(v)
    index_dict = {}
    for k, v in index_dict_.items():
        index_dict[k] = np.array(v)
        
    for k, v in sv_dict.items():
        ax1.plot(index_dict[k],sv_dict[k],'+',label='q='+str(k))
        ax2.bar(float(k), - np.sum(np.square(sv_dict[k]) * np.log(np.square(sv_dict[k]))) )
        
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_ylabel('sv')
    ax1.set_xlabel('#')
    ax2.set_ylabel('S')
    ax2.set_xlabel('q')
    ax1.grid()
    fig.suptitle('site='+str(l), fontsize=16)
    plt.show()
