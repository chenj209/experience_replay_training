# Correcting the mistake and completing the plotting code
import matplotlib.pyplot as plt
import numpy as np

# Redefine corrected data arrays
AE_all_train = np.clip(np.array([
    -14.7208493, -145.90918, -0.480210202, -299.325671, -0.358247895, -11.6893968, -16.6849746, 0.27193991,
    -0.420040774, -6.92803966, 0.280342747, 0.507875959, 0.641711362, 0.782494439, 0.851551699, 0.856970297,
    0.826425014, 0.753260989, 0.656059086, 0.644363899, 0.635113357, 0.623191091, 0.595737696, 0.597781287,
    0.600590526, 0.586582698, 0.546488357, 0.481116818, 0.479737225, 0.841126799
]), 0, None)

baseline_all = np.clip(np.array([
    -56.73179398, -14.99547361, -12.27629356, -88.15880465, -21.42692222, -20.09542764, -15.18650373, -4.61196149,
    -5.10279296, -8.7545838, -0.49138004, 0.47758235, 0.72183078, 0.80438312, 0.86586024, 0.86626945,
    0.82229827, 0.73537285, 0.60820037, 0.5815515, 0.55871302, 0.51136569, 0.44400158, 0.40031274,
    0.40171631, 0.37658428, 0.34243339, 0.29544911, 0.26090477, 0.77701724
]), 0, None)

loc_code_train = np.clip(np.array([
    -3.87095957, -16.93100911, -9.54284004, -145.50244959, -89.41382053, -19.45363051, -235.41195172, -9.84269803,
    -7.52832761, -3.46627707, -1.11046579, -4.01694596, 0.65143243, 0.78005672, 0.85085, 0.85371228,
    0.8151092, 0.73046537, 0.60822482, 0.59236467, 0.56944352, 0.52805334, 0.46880073, 0.41667069,
    0.41880289, 0.39415207, 0.35704433, 0.30859624, 0.27837399, 0.7762997
]), 0, None)

loc_code = np.clip(np.array([
    -3.79102452, -16.48415272, -9.9004209, -144.2789678, -88.11077942, -19.39633392, -231.90778006, -10.08065913,
    -7.36402465, -2.72494674, -0.78718217, -3.48723279, 0.6600384, 0.7921163, 0.86386683, 0.86730893,
    0.82857523, 0.74528364, 0.60842483, 0.5844956, 0.56184909, 0.51578705, 0.45426805, 0.39980055,
    0.404584, 0.38314808, 0.34787122, 0.30048113, 0.26856851, 0.77516429
]), 0, None)
vae_loc_code = np.clip(np.array([
    0.99636179, 0.99807854, 0.9968428 , 0.99596351, 0.9977631 ,
    0.99118992, 0.98319405, 0.99569177, 0.99441919, 0.99632987,
    0.98811238, 0.9002527 , 0.63962633, 0.77801874, 0.85984011,
    0.86673436, 0.83036163, 0.75041451, 0.62313295, 0.59555142,
    0.56946401, 0.52397292, 0.46216537, 0.40815683, 0.41473064,
    0.39402396, 0.35902311, 0.31171148, 0.27397025, 0.784737
]), 0, None)

vertical_levels = np.arange(1, 31)

plt.figure(figsize=(10, 8))

AE_all_train[AE_all_train<0]=0
loc_code_train[loc_code_train<0] = 0
loc_code[loc_code<0] = 0
baseline_all[baseline_all<0]=0

plt.plot(loc_code, vertical_levels, label="loc_code", marker='o', color='red')
plt.plot(loc_code_train, vertical_levels, label="loc_code_train", marker='x')
plt.plot(AE_all_train, vertical_levels, label="AE_all_train", marker='^')
plt.plot(baseline_all, vertical_levels, label="baseline_all", marker='s', color='blue')  # Add VAE_train in blue
plt.plot(vae_loc_code, vertical_levels, label="vae_loc_code", marker='o', color='purple')  # Add VAE_train in blue

plt.title("Model Performance Comparison in R2 for 30 Vertical Levels")
plt.ylabel("Vertical Levels")
plt.xlabel("R2 Score")
plt.gca().invert_yaxis()  # Invert y-axis to have level 1 at the top
plt.yticks(vertical_levels)
plt.legend()
plt.grid(True)
plt.show()
