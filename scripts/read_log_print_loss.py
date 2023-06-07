import json

file_path = "./log ori.json"

# read origin, resume log 
with open(file_path, 'r') as file:
    
    trainLossList = [0 for i in range(67)]
    validationLossList = [0 for i in range(67)]

    # origin log
    data = json.load(file)
    for dic in data:
        epoch = dic['epoch']
        trainLossList[epoch] += dic['main/loss']
        if 'validation/main/loss' in dic:
            validationLossList[epoch] = dic['validation/main/loss']

    for i in range(len(trainLossList)):
        trainLossList[i] /=14
    
    # resume log
    for j in range(45, 50):
        trainLossList[j] = 0
        validationLossList[j] = 0

    with open('./log resu.json', 'r') as file:
        data = json.load(file)
        for dic in data:
            epoch = dic['epoch'] + 45
            trainLossList[epoch] += dic['main/loss']
            if 'validation/main/loss' in dic:
                validationLossList[epoch] = dic['validation/main/loss']
            
        for i in range(45, 61):
            trainLossList[i] /=14

    for j in range(62):
        print(j, ": ", trainLossList[j])
    
    # valloss half
    # for i in range(1, 61):
    #     if validationLossList[i] < 0.01:
    #         validationLossList[i] = (validationLossList[i+1] + validationLossList[i-1]) / 2

    import matplotlib.pyplot as plt
    import numpy
    t = numpy.arange(1., 61., 1.)
    
    plt.plot(t, trainLossList[1:61], 'ro', label="test loss")
    plt.plot(t, validationLossList[1:61], 'bs', label="validation loss")

    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.legend()
    # plt.ylim(0.0, 0.08)
    plt.show()

