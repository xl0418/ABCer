# %%
from ABCer import ABCer
import numpy as np

# Model 1
def model1(para, time_survey=np.arange(18)):
    # time_survey = np.arange(18)
    y = para[0] * np.exp(para[1] * time_survey)
    return y

# The data of the coronavirus outbreak in NL from 27-2-2020 to 17-03-2020
observations = np.array([
    1.0, 7.0, 10.0, 24.0, 38.0, 82.0, 128.0, 188.0, 265.0, 321.0, 382.0, 503.0,
    614.0, 804.0, 959.0, 1135.0, 1413.0, 1705.0
])
time = np.arange(len(observations))

# Initialize the ABC approach
test_ABC1 = ABCer(iterations=50, particles=10000, observations=observations)
test_ABC1.initialize_model(model1)
test_ABC1.initialize_parameters([0.0, 1.0])

# Launch...
test_list1 = test_ABC1.ABC(prior_paras=[0.0, 1.0, 1.0, 2.0])


# %%
