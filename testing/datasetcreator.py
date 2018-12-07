import numpy as np
import pickle
from multiprocessing import Pool
import time

##### init
datasets = []
gamecount = 0

data = pickle.load(open('repfile_x_quan_1000.dat', 'rb'))

primary_map = 'Stadium_P'


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

def createsets(game):
    inputarray = []
    deltas = []
    answerarray = []
    carmetrics = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'boost', 'boost_active', 'boost_collect']
    ballmetrics = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']
    #print('in')

    for frame in range(1, game['frames'] - 2):
        xtemp = []
        ytemp = []
        dtemp = []
        goal = 0
        try:
            #add player data
            for player in game['players']:
                name1 = player
                player = game['players'][player]
                xtemp.extend(player['frame'].loc[frame, carmetrics])
                ytemp.extend(player['frame'].loc[frame + 1, carmetrics])

                dtemp.extend([xtemp[pos] - ytemp[pos] for pos in range(12)]) #add deltas

                if player['frame'].goal.iloc[frame] == True: #break off if there is a goal in the input
                    raise Exception('goal in input')
                if player['frame'].goal.iloc[frame + 1] == True: #record which team scored
                    if 'blue' in player['name']:
                        goal = 1
                    else:
                        goal = 2

            #add ball metrics
            xtemp.extend(game['ball'].loc[frame, ballmetrics])
            ytemp.extend(game['ball'].loc[frame + 1, ballmetrics])
            xtemp.extend([0, 0, 1])

            if goal == 0:
                ytemp.extend([0, 0, 1])
            elif goal == 1:
                ytemp.extend([0, 1, 0])

            elif goal == 2:
                ytemp.extend([0, 0, 1])

            #remove nan for zeros might not do this afterall
            inputarray.append(np.nan_to_num(xtemp))
            answerarray.append(np.nan_to_num(ytemp))
            deltas.append(np.array(dtemp))
        except:
            pass #this is going to kill me one day

    inputarray = np.array(inputarray)
    answerarray = np.array(answerarray)
    deltas = np.array(deltas)
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
            if repproto.game_metadata.team_size != 3:
                print('nontrip')
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
    p = Pool(processes=10)
    datasets = p.map(createsets, getgame())
    p.close()

    write_name = 'set1_' + 'yo '+ '.pset'
    pickle.dump(datasets, open(write_name, 'wb'), -1)

    print('done')