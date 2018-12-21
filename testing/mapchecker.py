import numpy as np
import pickle


##### init
maps = {}
gamecount = 0

names = []
for i in range(10):
    names.append('repfile_' + str(i) + '_quan_1000.dat')

for datfile in names:
    data = 0
    data = pickle.load(open(datfile, 'rb'))

    for replay in data:
        repproto = replay.get_proto()
        map_name = repproto.game_metadata.map
        if repproto.game_metadata.team_size != 3:
            pass
        else:
            try:
                check = maps[map_name]
                maps[map_name] += 1
            except:
                maps[map_name] = 1

print(maps)

'''
most popular map:
'Stadium_P'
    'stadium_day_p'
    'Stadium_Foggy_P'
    'Stadium_Winter_P'
    'stadium_foggy_p'
    'Stadium_p'
    'stadium_p'
'''