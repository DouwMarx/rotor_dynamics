import src.sensitivity_study as sens
import numpy as np

#sens.compute_grad_for_T0_change(save_name= "grad_change_with_T_rms.pdf")

I1 = 0.1  # kgm^2
I2 = 0.2  # kgm^2
c = 0.05  # Nms/rad
k = 2500  # Nm/rad

constants_dict = {"parameters_at_gradient_evaluation": np.array([I1, I2, c, k]),
                  "operating_conditions": {"T_0": 10}}
print(sens.compute_grad_for_sys(constants_dict))

# TODO2: Make gradient computable for both masses.