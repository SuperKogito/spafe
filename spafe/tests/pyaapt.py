import numpy as np

# border definitions of the 24 critical bands of hearing
bark = [100,   200,  300,  400,  510,  630,   770,   920, 
        1080, 1270, 1480, 1720, 2000, 2320,  2700,  3150,
        3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]

eq_loudness = np.array(
    [[ 55,   40,  32,  24,  19,  14, 10,  6,  4,  3,  2,  
        2,    0,  -2,  -5,  -4,   0,  5, 10, 14, 25, 35], 
     [ 66,   52,  43,  37,  32,  27, 23, 21, 20, 20, 20,  
       20,   19,  16,  13,  13,  18, 22, 25, 30, 40, 50], 
     [ 76,   64,  57,  51,  47,  43, 41, 41, 40, 40, 40,
     39.5, 38,  35,  33,  33,  35, 41, 46, 50, 60, 70], 
     [ 89,   79,  74,  70,  66,  63, 61, 60, 60, 60, 60,  
       59,   56,  53,  52,  53,  56, 61, 65, 70, 80, 90], 
     [103,   96,  92,  88,  85,  83, 81, 80, 80, 80, 80,  
       79,   76,  72,  70,  70,  75, 79, 83, 87, 95,105], 
     [118,  110, 107, 105, 103, 102,101,100,100,100,100,  
       99,   97,  94,  90,  90,  95,100,103,105,108,115]])

loudn_freq = np.array(
    [31.62,   50,  70.7,   100, 141.4,   200, 316.2,  500, 
     707.1, 1000,  1414,  1682,  2000,  2515,  3162, 3976,
     5000,  7071, 10000, 11890, 14140, 15500])

# calculate bark-filterbank
loudn_bark = np.zeros((eq_loudness.shape[0], len(bark)))

i = 0
j = 0

for bsi in bark:

    while j < len(loudn_freq) and bsi > loudn_freq[j]:
        j += 1
    
    j -= 1
    
    if np.where(loudn_freq == bsi)[0].size != 0: # loudness value for this frequency already exists
        loudn_bark[:,i] = eq_loudness[:,np.where(loudn_freq == bsi)][:,0,0]
    else:
        w1 = 1 / np.abs(loudn_freq[j] - bsi)
        w2 = 1 / np.abs(loudn_freq[j + 1] - bsi)
        loudn_bark[:,i] = (eq_loudness[:,j]*w1 + eq_loudness[:,j+1]*w2) / (w1 + w2)
    
    i += 1

