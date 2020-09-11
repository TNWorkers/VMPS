import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Plot the singular values of the MPS')
parser.add_argument('-dir', type=str, default='.', dest='dir', help='The directory where the singular values are stored. Files have to be named as sv_final_<N>.dat.')

args = parser.parse_args()

print('Plot files in directory',args.dir)
nums=[]
for f in os.listdir(args.dir):
    if f.find('sv_final_') != -1:
        nums.append(int(f[len('sv_final_'):f.find('.dat')]))

for l in nums:
    fig, (ax1, ax2) = plt.subplots(2, 1)
    name = args.dir + '/sv_final_' + str(l) + '.dat'
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
        qval = q
            
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
    svs = np.array(svs_,dtype=float)
    
    sv_dict = {}
    for k, v in sv_dict_.items():
        sv_dict[k] = np.array(v)
    index_dict = {}
    for k, v in index_dict_.items():
        index_dict[k] = np.array(v)

    pos=0.
    for k, v in sv_dict.items():
        ax1.plot(index_dict[k],sv_dict[k],'+',label='q='+k)
        ax2.bar(pos, - np.sum(np.square(sv_dict[k]) * np.log(np.square(sv_dict[k]))) )
        pos += 1
        
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=6, mode="expand", borderaxespad=0.)
    ax1.set_yscale('log')
    ax1.set_ylabel('sv')
    ax1.set_xlabel('#')
    ax2.set_ylabel('S')
    ax2.set_xlabel('q')
    ax1.grid()
    fig.suptitle('site='+str(l), fontsize=16)
    plt.show()
