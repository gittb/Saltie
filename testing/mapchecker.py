import numpy as np
import pickle


##### init
maps = {}
gamecount = 0

data = pickle.load(open('repfile_x_quan_1000.dat', 'rb'))

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