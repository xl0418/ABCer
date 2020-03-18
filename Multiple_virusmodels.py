# %%
from ABCer import ABCer
import matplotlib.pyplot as plt
import numpy as np


# Model 1
def model1(para, time_survey=np.arange(18)):
    # time_survey = np.arange(18)
    y = para[0] * np.exp(para[1] * time_survey)
    return y


observations = np.array([
    1.0, 7.0, 10.0, 24.0, 38.0, 82.0, 128.0, 188.0, 265.0, 321.0, 382.0, 503.0,
    614.0, 804.0, 959.0, 1135.0, 1413.0, 1705.0
])
time = np.arange(len(observations))

test_ABC1 = ABCer(50, 10000, observations=observations)
test_ABC1.initialize_model(model1)
test_ABC1.initialize_parameters([0.0, 1.0])
test_list1 = test_ABC1.ABC(prior_paras=[0.0, 1.0, 1.0, 2.0])

plt.plot(time, observations, 'o')
para_inferred = []
para_inferred.append(np.mean(test_list1[0][20, :]))
para_inferred.append(np.mean(test_list1[1][20, :]))
extend_time = np.arange(21)
y_inferred = model1(para_inferred, np.arange(21))

plt.plot(extend_time, y_inferred, 'x', color='r')
plt.xlabel("Days")
plt.ylabel('Number of infected cases')

# %%
# Model 2
def model2(para, time_survey=np.arange(18)):
    # time_survey = np.arange(18)
    y = para[0] * np.exp(para[1] * time_survey**2 + para[2] * time_survey)
    return y


observations = np.array([
    1.0, 7.0, 10.0, 24.0, 38.0, 82.0, 128.0, 188.0, 265.0, 321.0, 382.0, 503.0,
    614.0, 804.0, 959.0, 1135.0, 1413.0, 1705.0
])
time = np.arange(len(observations))

test_ABC2 = ABCer(100, 10000, observations=observations)
test_ABC2.initialize_model(model2)
test_ABC2.initialize_parameters([0.0, 0.1, 0.1])
test_list2 = test_ABC2.ABC(prior_paras=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

plt.plot(time, observations, 'o')
para_inferred = []
para_inferred.append(np.mean(test_list2[0][20, :]))
para_inferred.append(np.mean(test_list2[1][20, :]))
para_inferred.append(np.mean(test_list2[2][20, :]))

extend_time = np.arange(18)
y_inferred = model2(para_inferred, np.arange(18))

plt.plot(extend_time, y_inferred, 'x', color='r')


# %%
# Model 3
def model3(para, time_survey=np.arange(18)):
    # time_survey = np.arange(18)
    y = para[0] * time_survey**2 + para[1] * time_survey + para[2]
    return y


observations = np.array([
    1.0, 7.0, 10.0, 24.0, 38.0, 82.0, 128.0, 188.0, 265.0, 321.0, 382.0, 503.0,
    614.0, 804.0, 959.0, 1135.0, 1413.0, 1705.0
])
time = np.arange(len(observations))

test_ABC3 = ABCer(100, 10000, observations=observations)
test_ABC3.initialize_model(model3)
test_ABC3.initialize_parameters([0.0, 1.0, 1.0])
test_list3 = test_ABC3.ABC(prior_paras=[0.0, 10.0, 1.0, 2.0, 1.0, 2.0])

plt.plot(time, observations, 'o')
para_inferred = []
para_inferred.append(np.mean(test_list3[0][20, :]))
para_inferred.append(np.mean(test_list3[1][20, :]))
para_inferred.append(np.mean(test_list3[2][20, :]))

extend_time = np.arange(18)
y_inferred = model3(para_inferred, np.arange(18))

plt.plot(extend_time, y_inferred, 'x', color='r')


# %%
# Model 4
def model4(para, time_survey=np.arange(18)):
    # time_survey = np.arange(18)
    y = para[0] * time_survey**3 + para[1] * time_survey**2 + para[2] * time_survey + para[3]
    return y


observations = np.array([
    1.0, 7.0, 10.0, 24.0, 38.0, 82.0, 128.0, 188.0, 265.0, 321.0, 382.0, 503.0,
    614.0, 804.0, 959.0, 1135.0, 1413.0, 1705.0
])
time = np.arange(len(observations))

test_ABC4 = ABCer(50, 10000, observations=observations)
test_ABC4.initialize_model(model3)
test_ABC4.initialize_parameters([0.0, 1.0, 1.0, 1.0])
test_list4 = test_ABC4.ABC(prior_paras=[0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])

plt.plot(time, observations, 'o')
para_inferred = []
para_inferred.append(np.mean(test_list4[0][20, :]))
para_inferred.append(np.mean(test_list4[1][20, :]))
para_inferred.append(np.mean(test_list4[2][20, :]))
para_inferred.append(np.mean(test_list4[3][20, :]))

extend_time = np.arange(18)
y_inferred = model4(para_inferred, np.arange(18))

plt.plot(extend_time, y_inferred, 'x', color='r')


# %%
