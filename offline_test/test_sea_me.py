# Correcting the mistake and completing the plotting code
import matplotlib.pyplot as plt
import numpy as np

# Redefine corrected data arrays
vae = np.array([
    0.00056181,  0.00138204, -0.00060969, -0.00167266,  0.00117735,
            0.00025447,  0.00143087,  0.0019285 , -0.00135371, -0.00260053,                                                                                                                 0.00067258, -0.00109831,  0.00218768,  0.0017137 ,  0.00057403,                                                                                                                -0.00345896, -0.00427192, -0.00124799, -0.00078442,  0.0206292 ,                                                                                                                 0.03194723,  0.02027893,  0.02298069, -0.00048192, -0.02757675,
                   -0.00368799,  0.0169141 ,  0.01871832,  0.00291817, -0.00825924
        ])

vae3 = np.array([-0.00432318,  0.00771181, -0.0001112 ,  0.00228962, -0.00156958,
                        -0.00096621,  0.00458431,  0.00123213,  0.00228928, -0.00068882,
                        -0.00168983,  0.00276163, -0.00218424,  0.00127307, -0.00257799,
                         0.00078469,  0.00686913,  0.02196395,  0.00214615,  0.00390858,
                        -0.00423984,  0.00812768,  0.02845794,  0.04960195,  0.04178799,
                         0.03386126,  0.03066268,  0.05506621, -0.00939906, -0.07913181])
vae4 = np.array([0.00265987,  0.00279153, -0.00024077, -0.00225084,  0.0009382 ,
                        -0.00318139, -0.00578015,  0.0008705 ,  0.00763699, -0.00197509,                                                                                                                                                                             0.00309825,  0.00666571,  0.0037522 , -0.0007544 ,  0.01098767,                                                                                                                                                                             0.00875682,  0.01637888, -0.0012112 , -0.01796983,  0.01126212,
                         0.01455286,  0.04299984,  0.00113535, -0.00845061,  0.01163414,                                                                                                                                                                             0.02066069, -0.02678637,  0.02457376,  0.06819238,  0.03647877])
vae4_rlvl = np.array([-0.00332141, -0.00456592,  0.00642823, -0.00169289,  0.00104802,
                              0.00389263,  0.00481032,  0.01300046,  0.00444966,  0.00195494,
                             -0.00551142, -0.00064019, -0.00515085, -0.00310076,  0.00634779,                                                                                                                 0.00231351, -0.00048268, -0.00032833, -0.01894645, -0.02220264,                                                                                                                -0.01297528,  0.01860208,  0.00120986, -0.03735273, -0.00780937,                                                                                                                 0.02089283,  0.02528955,  0.00072597, -0.01346664,  0.00226706])

baseline2 = np.array([
     8.1345800e-04,  1.4795247e-05,  4.0487156e-04,  2.7833616e-03,
             4.3079844e-03,  1.6077952e-03, -1.9965840e-03,  5.1671318e-03,
                     2.7281374e-03,  1.7563744e-04, -4.3153861e-03,  3.7927978e-04,
                            -1.7080299e-03,  5.5790041e-03,  8.2003577e-03, -4.0723286e-03,
                                    1.1870065e-02,  1.6105345e-02, -5.1091041e-04, -8.0885151e-03,
                                           -3.9451797e-02,  3.8751695e-02,  4.8204321e-02, -1.9312961e-02,
                                                  -4.1992101e-03,  3.6639634e-03,  6.0268123e-02,  4.0160652e-02,
                                                          1.9750638e-02, -7.9817951e-02
])

baseline = np.array([
    -7.77134486e-03,  6.33011619e-03,  6.13756338e-03,  2.17349618e-03,
            2.29123607e-03, -7.53706787e-04,  9.87763237e-03, -2.26465100e-03,
                   -3.37055062e-05,  3.38653289e-03,  5.91836171e-03,  6.33453019e-03,
                          -4.28030122e-04, -7.10271159e-03, -2.86122854e-03, -5.30467695e-03,
                                 -6.58001844e-03, -3.88799980e-02, -5.24060205e-02, -1.22226164e-01,
                                        -1.17369100e-01, -2.77977325e-02,  1.94033645e-02,  5.73885068e-02,
                                                1.47659317e-01,  1.06860057e-01,  1.00342356e-01,  7.77206868e-02,
                                                        3.96828093e-02,  5.97661585e-02
])

vertical_levels = np.arange(1, 31)
vertical_levels_ticks = np.array([
    3.64346569,   7.59481965,  14.35663225,  24.61222,     38.26829977,
      54.59547974,  72.01245055,  87.82123029, 103.31712663, 121.54724076,
       142.99403876, 168.22507977, 197.9080867,  232.82861896, 273.91081676,
        322.24190235, 379.10090387, 445.9925741,  524.68717471, 609.77869481,
         691.38943031, 763.40448111, 820.85836865, 859.53476653, 887.02024892,
          912.64454694, 936.19839847, 957.48547954, 976.32540739, 992.55609512]).astype(int)
vertical_levels_ticks = [str(v) for v in vertical_levels_ticks]

plt.figure(figsize=(10, 8))
plot_lvl = 16

#baseline[baseline<0]=0
#baseline2[baseline2<0]=0
#vae[vae<0] = 0
#vae2[vae2<0] = 0

#plt.plot(baseline2, vertical_levels, label="baseline2", marker='s', color='red')  # Add VAE_train in blue
plt.plot(baseline[plot_lvl:], vertical_levels[plot_lvl:], label="baseline", marker='s', color='blue')  # Add VAE_train in blue
plt.plot(baseline2[plot_lvl:], vertical_levels[plot_lvl:], label="baseline+new_vars", marker='s', color='red')  # Add VAE_train in blue
plt.plot(vae[plot_lvl:], vertical_levels[plot_lvl:], label="vae_pre50", marker='o', color='purple')  # Add VAE_train in blue
#plt.plot(vae3, vertical_levels, label="vae_loc_code3", marker='o', color='orange')  # Add VAE_train in blue
plt.plot(vae4[plot_lvl:], vertical_levels[plot_lvl:], label="vae_pre382", marker='o', color='black')  # Add VAE_train in blue
plt.plot(vae4_rlvl[plot_lvl:], vertical_levels[plot_lvl:], label="vae_pre982", marker='o', color='green')  # Add VAE_train in blue
#plt.plot(vae2, vertical_levels, label="vae_loc_code2", marker='o', color='green')  # Add VAE_train in blue
plt.yticks(vertical_levels[plot_lvl:], vertical_levels_ticks[plot_lvl:])

plt.title("Model Performance Comparison in Mean Error for 30 Vertical Levels")
plt.ylabel("Vertical Levels(hPa)")
plt.xlabel("Mean error (standardized)")
plt.gca().invert_yaxis()  # Invert y-axis to have level 1 at the top
plt.legend()
plt.grid(True)
plt.show()
