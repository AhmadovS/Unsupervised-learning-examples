import matplotlib.pyplot as plt
import numpy as np
import tables

def display_fields(F, n_fields=24, D=28):
    for i in xrange(n_fields):
        plt.subplot(3, 8, i+1)
        plt.axis('off'); gray()
        plt.imshow(F[i].reshape( (D,D)), interpolation='nearest')
    plt.savefig('result.png')


result_location = "../mnits/output2/bs100-lr13-si2-spl10-sbn-sbn-200-200-10.2017-04-13-18-53/result.h5"
hparams = {}
with tables.openFile(result_location, "r") as h5:
    Lp_10 = h5.root.Lp_10[:]
    Lp_25 = h5.root.Lp_25[:]
    Lp_100 = h5.root.Lp_100[:]
    Lp_500 = h5.root.Lp_100[:]

    mparams = {
        'P_a'  : h5.root.P_a[-1,:],
        'P_b'  : h5.root.P_b[-1,:],
        'P_W'  : h5.root.P_W[-1,:,:],
        'Q_c'  : h5.root.Q_c[-1,:],
        'Q_b'  : h5.root.Q_b[-1,:],
        'Q_W'  : h5.root.Q_W[-1,:,:],
        'Q_V'  : h5.root.Q_V[-1,:,:],
        'Q_Ub' : h5.root.Q_Ub[-1,:,:],
        'Q_Uc' : h5.root.Q_Uc[-1,:,:],
    }

# model = hparams['model']
# model.set_model_params(mparams)
# n_hid, n_vis = mparams['P_W'].shape
# assert n_vis == model.n_vis
# assert n_hid == model.n_hid

plt.clf()
plt.title("est. Log-Likelihood");
plt.xlabel("Epoch")
for var, label in ( (Lp_10, 'L_{10}'), (Lp_25, 'L_{25}'), (Lp_100, 'L_{100}'),):
    plt.plot(var[:], label="$%s$ =  %5.2f"%(label, var[-1])); 

legend(loc='lower right')
plt.savefig('ll.png')
plt.clf()

display_fields(mparams.P_W)
plt.savefig('p_w.png')