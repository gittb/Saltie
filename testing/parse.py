from framework.replay.replay_format import GeneratedReplay
import numpy as np
import pandas as pd
import pickle
import time

data = pickle.load(open('repfile_x_quan_1000.dat', 'rb'))
all_cat = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'throttle', 'steer', 'handbrake', 'ball_cam', 'dodge_active', 'double_jump_active', 'jump_active', 'boost', 'boost_active', 'ping', 'boost_collect']
ball_cat = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'hit_team_no']
game_cat = ['time', 'delta', 'seconds_remaining', 'replicated_seconds_remaining', 'ball_has_been_hit', 'goal_number']
'''
parsedrep = data[0]

repdat = parsedrep.get_pandas()
repproto = parsedrep.get_proto()

#print([i for i in repdat])
multidex = list(repdat.columns.get_level_values(0).unique())
print(repproto.game_metadata.demos)

print(repdat['ball'][ball_cat].values[:10])
print(repdat['game'][game_cat].values[:10])

fuckit = []

#print(repproto)
sortednames = [[x.name, x.id.id, x.is_orange] for x in repproto.players] #should do dict
print(sortednames)
sortednames.sort(key=lambda x: x[2])
'''
doubleparse = []

for replay in data:
    try:
        repdat = replay.get_pandas()
        repproto = replay.get_proto()
        if repproto.game_metadata.team_size != 3:
            raise Exception('not a trip game')
        #name handeling shit
        sortednames = [[x.name, x.id.id, x.is_orange] for x in repproto.players]  # should do dict
        sortednames.sort(key=lambda x: x[2])

        names_id = {}
        names_name = {}
        name_niks = {}
        count = 0
        for i in sortednames:
            names_id[i[1]] = [i[0], i[2]]
            names_name[i[0]] = [i[1], i[2]]
            count += 1

        #all dict creawtion

        game = {}
        players = {}

        # gamedata
        game['kickoffs'] = [[x.start_frame_number, x.end_frame_number] for x in repproto.game_stats.kickoffs]
        game['teamsize'] = repproto.game_metadata.team_size
        game['frames'] = repproto.game_metadata.frames
        game['map'] = repproto.game_metadata.map
        game['ball'] = repdat['ball'][ball_cat].values
        game['misc'] = repdat['game'][game_cat].values


        #get plays atribs
        count = 1

        #new frames 23 = hit | 24 = demo | 25 = goal | 26 = collision

        for i in names_name:
            # gen player numpy
            playerarray = repdat[i][all_cat].values
            axislen = playerarray.shape[0]
            concatarray = np.zeros((axislen, 3))
            playerarray = np.concatenate((playerarray, concatarray), 1)

            # name creation
            if names_name[i][1] == 1:
                name = 'orange' + str(count)
            else:
                name = 'blue' + str(count)
            name_niks[names_name[i][0]] = name
            # internal dict
            inner = {
                'vals': playerarray,
                'name': i,
                'team': names_name[i][1],
                'id': names_name[i][0]
            }
            players[name] = inner
            if count == 3:
                count = 1
            else:
                count += 1

        #additional player vals
        goalframes = [[x.frame_number, x.player_id.id] for x in repproto.game_metadata.goals]
        demoframes = [[x.frame_number, x.attacker_id.id] for x in repproto.game_metadata.demos]
        hits = [[x.frame_number, x.player_id.id] for x in repproto.game_stats.hits]

        # new frames 23 = hit | 24 = demo | 25 = goal | 26 = collision (not added yet)
        #print('\n\n goals \n\n')
        for goal in goalframes:
            frame = goal[0]
            players[name_niks[goal[1]]]['vals'][frame-2][25] = 1 #only god understands this indexing

        #print('\n\n demos \n\n')
        for demo in demoframes:
            frame = demo[0]
            players[name_niks[demo[1]]]['vals'][frame-2][24] = 1

        #print('\n\n hits \n\n')
        for hit in hits:
            frame = hit[0]
            players[name_niks[hit[1]]]['vals'][frame-2][23] = 1

        game['players'] = players

        #print('\n\n finalprint \n\n')
        #print(game.keys())
        doubleparse.append(game)
        #print(players['blue1']['vals'][500:520], frame)
        #time.sleep(3)
    except:
        print('error')
        print(len(doubleparse))

pickle.dump(doubleparse, open('dub1.dat', 'wb'))

print(len(doubleparse))
print('done')