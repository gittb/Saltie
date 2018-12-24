import numpy as np
import pickle
import pandas as pd
from multiprocessing import Pool
import time

##### init
datasets = []
gamecount = 0

data = pickle.load(open('repfile_1_quan_1000.dat', 'rb'))

namelist = ['Stadium_P', 'stadium_day_p', 'Stadium_Foggy_P', 'Stadium_Winter_P', 'stadium_foggy_p', 'Stadium_p', 'stadium_p']


#            0          1       2         3       4         5       6       7        8          9           10              11          12        13        14          15          16                  17                  18          19          20            21           22
all_cat = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'throttle', 'steer', 'handbrake', 'ball_cam', 'dodge_active', 'double_jump_active', 'jump_active', 'boost', 'boost_active', 'ping', 'boost_collect']
ball_cat = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'hit_team_no']
game_cat = ['time', 'delta', 'seconds_remaining', 'replicated_seconds_remaining', 'ball_has_been_hit', 'goal_number']


class DSgame():

    def __init__(self, inputarray, deltas, answerarray, game_map):
        self.inputarray = inputarray
        self.deltas = deltas
        self.answerarray = answerarray
        self.map = map

def onehotgen(val, cases):
    if val == True:
        val = 1
    elif val == False:
        val = 0
    onehot = np.zeros((cases), dtype=np.float64)
    onehot[val] = 1
    return onehot


def createsets(game):
    inputarray = [] #go ahead and allocate full numpy array
    deltas = [] #go ahead and allocate full numpy array
    answerarray = [] #go ahead and allocate full numpy array

    carmetrics = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'boost', 'boost_active', 'boost_collect']
    ballmetrics = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']
    #print('in')

    for frame in range(1, game['frames'] - 2):
        xtemp = [] #allocate as well
        ytemp = []#allocate as well
        dtemp = []#allocate as wellK
        goal = 0
        try:
            #add player data
            cur = 0
            for player in game['players']:
                name1 = player
                player = game['players'][player]
                xprecheck = player['frame'].loc[frame, carmetrics]
                yprecheck = player['frame'].loc[frame + 1, carmetrics]
                if xprecheck[13]:
                    xprecheck[13] = 1
                else:
                    xprecheck[13] = 0

                if pd.isnull(xprecheck[14]):
                    xprecheck[14] = 0

                if yprecheck[13]:
                    yprecheck[13] = 1
                else:
                    yprecheck[13] = 0

                if pd.isnull(yprecheck[14]):
                    yprecheck[14] = 0
                xtemp.extend(xprecheck)
                ytemp.extend(yprecheck)
                dtemp.extend([yprecheck[pos] - xprecheck[pos] for pos in range(12)]) #add deltas
                if player['frame'].goal.iloc[frame] == True: #break off if there is a goal in the input
                    raise Exception('goal in input')
                if player['frame'].goal.iloc[frame + 1] == True: #record which team scored
                    if 'blue' in player['name']:
                        goal = 1
                    else:
                        goal = 2

            #add ball metrics
            btemp = game['ball'].loc[frame, ballmetrics]
            btemp2 = game['ball'].loc[frame + 1, ballmetrics]
            dtemp.extend([btemp2[pos] - btemp[pos] for pos in range(12)])

            xtemp.extend(btemp)
            ytemp.extend(btemp2)
            xtemp.extend([0, 0, 1])

            ytemp.extend(onehotgen(goal,3))
            '''
            #replaced by the onehotgen
            if goal == 0:
                ytemp.extend([0, 0, 1])
            elif goal == 1:
                ytemp.extend([0, 1, 0])
    
            elif goal == 2:
                ytemp.extend([0, 0, 1])
            '''

            if sum(dtemp) == 0 or pd.isnull(sum(dtemp)):
                raise Exception('no change in delta')
            else:
                pass
            inputarray.append(np.nan_to_num(xtemp))
            answerarray.append(np.nan_to_num(ytemp))
            deltas.append(np.array(dtemp))

        except:
            pass #this is going to kill me one day

    inputarray = np.array(inputarray, dtype=np.float64)
    answerarray = np.array(answerarray, dtype=np.float64)
    deltas = np.array(deltas, dtype=np.float64)
    #print(inputarray.shape, answerarray.shape, deltas.shape)
    #print(inputarray[548], answerarray[548], deltas[548])
    game_map = game['map']
    gameobj = DSgame(inputarray, answerarray, deltas, game_map)
    return gameobj

def getgame():
    for replay in data:
        repdat = replay.get_pandas()
        repproto = replay.get_proto()

        try:
            repdat = replay.get_pandas()
            repproto = replay.get_proto()
            #if repproto.game_metadata.team_size != 3 or repproto.game_metadata.map not in namelist:
            if repproto.game_metadata.team_size != 3:
                print('wrongmap')
            elif repproto.game_metadata.map not in namelist:
                print('wrongmap')
            else:
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
                game['ball'] = repdat['ball'][ball_cat]
                game['misc'] = repdat['game'][game_cat]



                #get plays atribs
                count = 1

                #new frames 23 = hit | 24 = demo | 25 = goal | 26 = collision

                for i in names_name:
                    # gen player numpy
                    playerarray = repdat[i][all_cat]
                    playerarray['goal'] = False
                    playerarray['demo'] = False
                    playerarray['hit'] = False

                    # name creation
                    if names_name[i][1] == 1:
                        name = 'orange' + str(count)
                    else:
                        name = 'blue' + str(count)
                    name_niks[names_name[i][0]] = name
                    # internal dict
                    inner = {
                        'frame': playerarray,
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

                # new frames 23 = hit | 24 = demo | 25 = goal
                #print('\n\n goals \n\n')
                for goal in goalframes:
                    frame = goal[0]
                    players[name_niks[goal[1]]]['frame'].goal.iloc[frame - 2] = True #only god understands this indexing
                    #print(players[name_niks[goal[1]]]['frame'].iloc[frame - 2])

                #print('\n\n demos \n\n')
                for demo in demoframes:
                    frame = demo[0]
                    players[name_niks[demo[1]]]['frame'].demo.iloc[frame-2] = True
                    #print(players[name_niks[goal[1]]]['frame'].iloc[frame - 2])

                #print('\n\n hits \n\n')
                for hit in hits:
                    frame = hit[0]
                    players[name_niks[demo[1]]]['frame'].hit.iloc[frame - 2] = True
                    #print(players[name_niks[goal[1]]]['frame'].iloc[frame - 2])




                game['players'] = players


                yield game
        except:
            print('shitsbroke')


if __name__ == '__main__':
    p = Pool(processes=2)
    datasets = p.map(createsets, getgame())
    p.close()

    write_name = 'set1_' + 'yo '+ '.pset'
    pickle.dump(datasets, open(write_name, 'wb'), -1)

    print('done')
