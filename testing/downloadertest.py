from framework.data_generator.replay.replay_generator import ReplayDownloaderGenerator
import time
import sys
import pickle


downloader = ReplayDownloaderGenerator()
downloader.initialize(buffer_size=10, parallel_threads=1)

numround = 20
numitems = 1000
datalist = []
check = False

for round1 in range(numround):

    datalist = []

    for epoch in range(numitems):
        if epoch % 100 == 0: print(epoch)
        try:
            datalist.append(next(downloader.get_data()))
            if check == True:
                print('success moving on')
                check = False
            time.sleep(0.35)

        except:
            print('connection failed.. resetting')
            time.sleep(600)
            check = True
            epoch = epoch - 1
            downloader = ReplayDownloaderGenerator()
            downloader.initialize(buffer_size=10, parallel_threads=1)

    name = 'repfile2_' + str(round1) + "_quan_" + str(numitems) + ".dat"
    pickle.dump(datalist, open(name, 'wb'))
    print("wrote", name)