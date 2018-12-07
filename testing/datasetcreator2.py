from framework.replay.replay_format import GeneratedReplay
import numpy as np
import pandas as pd
import pickle
import time

##### init
datasets = []
gamecount = 0

data = pickle.load(open('repfile_x_quan_1000.dat', 'rb'))


#            0          1       2         3       4         5       6       7        8          9           10              11          12        13        14          15          16                  17                  18          19          20            21           22
all_cat = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'throttle', 'steer', 'handbrake', 'ball_cam', 'dodge_active', 'double_jump_active', 'jump_active', 'boost', 'boost_active', 'ping', 'boost_collect']
ball_cat = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'hit_team_no']
game_cat = ['time', 'delta', 'seconds_remaining', 'replicated_seconds_remaining', 'ball_has_been_hit', 'goal_number']

######

T_r = -36.07956616966136  # torque coefficient for roll
T_p = -12.14599781908070  # torque coefficient for pitch
T_y = 8.91962804287785  # torque coefficient for yaw
D_r = -4.47166302201591  # drag coefficient for roll
D_p = -2.798194258050845  # drag coefficient for pitch
D_y = -1.886491900437232  # drag coefficient for yaw


def predict_user_inputs(ang_vels, rotations, deltas):
    delta_omega = -ang_vels.diff(-1)
    delta_omega = delta_omega[(delta_omega.T != 0).any()]  # Filter all-zero rows.
    tau = delta_omega.divide(deltas[delta_omega.index], axis='index')
    tau_vectors = np.expand_dims(tau.values, 2)

    rotation_matrices = get_rotation_matrices(rotations.loc[delta_omega.index])
    inverse_rotation_matrices: pd.Series = np.transpose(rotation_matrices)
    inverse_rotation_array = np.stack(inverse_rotation_matrices.values)

    tau_local = np.matmul(inverse_rotation_array, tau_vectors)

    ang_vel_vectors = np.expand_dims(ang_vels.loc[delta_omega.index].values, 2)
    omega_local = np.matmul(inverse_rotation_array, ang_vel_vectors)

    omega_and_tau_locals = np.concatenate([omega_local, tau_local], axis=1)

    rhs_and_omega = np.apply_along_axis(get_rhs_and_omega, 1, omega_and_tau_locals)

    u = np.apply_along_axis(get_u, 1, rhs_and_omega)
    controls_data_frame = pd.DataFrame(data=np.squeeze(u, 2),
                                       index=delta_omega.index,
                                       columns=['predicted_input_roll', 'predicted_input_pitch', 'predicted_input_yaw'])

    controls_data_frame.clip(-1, 1, inplace=True)
    return controls_data_frame


def get_rotation_matrices(rotations: pd.Series) -> pd.Series:
    pitch = rotations.rot_x
    yaw = rotations.rot_y
    roll = rotations.rot_z

    cos_roll = np.cos(roll).rename('cos_roll')
    sin_roll = np.sin(roll).rename('sin_roll')
    cos_pitch = np.cos(pitch).rename('cos_pitch')
    sin_pitch = np.sin(pitch).rename('sin_pitch')
    cos_yaw = np.cos(yaw).rename('cos_yaw')
    sin_yaw = np.sin(yaw).rename('sin_yaw')

    components: pd.DataFrame = pd.concat([cos_roll, sin_roll, cos_pitch, sin_pitch, cos_yaw, sin_yaw], axis=1)

    rotation_matrix = components.apply(get_rotation_matrix_from_row, axis=1, result_type='reduce')
    return rotation_matrix


def get_rotation_matrix_from_row(components: pd.Series) -> np.array:
    cos_roll, sin_roll, cos_pitch, sin_pitch, cos_yaw, sin_yaw = components.values
    rotation_matrix = np.array(
        [[cos_pitch * cos_yaw, cos_yaw * sin_pitch * sin_roll - cos_roll * sin_yaw,
          -cos_roll * cos_yaw * sin_pitch - sin_roll * sin_yaw],
         [cos_pitch * sin_yaw, sin_yaw * sin_pitch * sin_roll + cos_roll * cos_yaw,
          -cos_roll * sin_yaw * sin_pitch + sin_roll * cos_yaw],
         [sin_pitch, -cos_pitch * sin_roll, cos_pitch * cos_roll]])
    return rotation_matrix


def get_rhs_and_omega(omega_and_tau):
    omega = omega_and_tau[:3]
    tau = omega_and_tau[3:]
    return np.array([
        tau[0] - D_r * omega[0],
        tau[1] - D_p * omega[1],
        tau[2] - D_y * omega[2],
        omega[0],
        omega[1],
        omega[2]
    ])


def get_u(rhs_and_omega):
    rhs = rhs_and_omega[:3]
    omega = rhs_and_omega[3:]
    return np.array([
        rhs[0] / T_r,
        rhs[1] / (T_p + np.sign(rhs[1]) * omega[1] * D_p),
        rhs[2] / (T_y - np.sign(rhs[2]) * omega[2] * D_y)
    ])


def get_controls(game):
    control_dict = {}
    for player in game['players'].keys():
        p_name = player
        player = game['players'][player]
        throttle = player['frame'].throttle / 128 - 1
        steer = -(player['frame'].steer / 128 - 1)
        _jump = player['frame'].jump_active % 2 == 1
        _double_jump_active = player['frame'].double_jump_active % 2 == 1
        _dodge_active = player['frame'].dodge_active % 2 == 1
        jump = _jump | _double_jump_active | _dodge_active
        boost = player['frame'].boost_active
        handbrake = player['frame'].handbrake

        frames_not_on_ground = player['frame'].loc[:, 'pos_z'][player['frame'].loc[:, 'pos_z'] > 18].index.values
        # print(frames_not_on_ground)
        rotations = player['frame'].loc[frames_not_on_ground, ['rot_x', 'rot_y', 'rot_z']]
        ang_vels = player['frame'].loc[frames_not_on_ground, ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']] / 1000

        predicted_inputs = predict_user_inputs(ang_vels, rotations, game['misc'].delta)
        # print(predicted_inputs)
        pitch = predicted_inputs.loc[:, 'predicted_input_pitch']
        yaw = predicted_inputs.loc[:, 'predicted_input_yaw']
        roll = predicted_inputs.loc[:, 'predicted_input_roll']

        # rotations = pd.concat((player.data.pos_z, player.data.loc[frames_not_on_ground, 'rot_x':'rot_z'],
        #                        predicted_inputs), axis=1)

        control_dict[p_name] = pd.DataFrame.from_dict({'throttle': throttle, 'steer': steer, 'pitch': pitch, 'yaw': yaw,
                                                  'roll': roll, 'jump': jump, 'boost': boost,
                                                  'handbrake': handbrake})

    return control_dict

def controltoinput(cdf, frame): #current work 12/5
    #['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake']
    #print(np.isnan(cdf.throttle.iloc[frame]))
    frame = frame - 1
    if np.isnan(cdf.throttle.iloc[frame]):
        return False
    else:
        hot = [1,0]
        cold = [0,1]
        temp = []
        temp.append(cdf.throttle.iloc[frame])
        if np.isnan(cdf.steer.iloc[frame]):
            temp.append(0)
        else:
            temp.append(cdf.steer.iloc[frame])

        if np.isnan(cdf.pitch.iloc[frame]):
            temp.append(0)
        else:
            temp.append(cdf.pitch.iloc[frame])

        if np.isnan(cdf.yaw.iloc[frame]):
            temp.append(0)
        else:
            temp.append(cdf.yaw.iloc[frame])

        if np.isnan(cdf.roll.iloc[frame]):
            temp.append(0)
        else:
            temp.append(cdf.roll.iloc[frame])

        if cdf.jump.iloc[frame] == False:
            temp.extend(cold)
        else:
            temp.extend(hot)

        if cdf.boost.iloc[frame] == False:
            temp.extend(cold)
        else:
            temp.extend(hot)

        if cdf.handbrake.iloc[frame] == False:
            temp.extend(cold)
        else:
            temp.extend(hot)
        return temp

def createsets(game):
    sets = []
    goals = 0
    control_df = get_controls(game)

    for frame in range(1, game['frames'] - 2):
        try:
            xtemp = []
            controltemp = []
            goal1 = 0
            for player in game['players']:
                name1 = player
                player = game['players'][player]
                xtemp.extend(player['frame'].loc[frame, ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']])
                if player['frame'].goal.iloc[frame] == True:
                    raise Exception('goal in x')
                test = controltoinput(control_df[name1], frame)
                if test != False:
                    controltemp.extend(test)
                else:
                    raise Exception('control failure')

            xtemp.extend(game['ball'].loc[frame, ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']])
            if goal1 == 0:
                xtemp.extend([0,0,1])
            elif goal1 == 1:
                xtemp.extend([0, 1, 0])
            elif goal1 == 2:
                xtemp.extend([0, 0, 1])
            #then add the control inputs for each player using control df
            xtemp.extend(controltemp)

            ytemp = []
            goal = 0
            for player in game['players']:
                name1 = player
                player = game['players'][player]
                ytemp.extend(player['frame'].loc[frame + 1, ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z',
                                         'ang_vel_x', 'ang_vel_y', 'ang_vel_z']])
                if player['frame'].goal.iloc[frame + 1] == True:
                    if 'blue' in player['name']:
                        goal = 1
                    else:
                        goal = 2
            ytemp.extend(game['ball'].loc[
                             frame + 1, ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'vel_x', 'vel_y', 'vel_z',
                                     'ang_vel_x', 'ang_vel_y', 'ang_vel_z']])
            if goal == 0:
                ytemp.extend([0, 0, 1])
            elif goal == 1:
                ytemp.extend([0, 1, 0])

            elif goal == 2:
                ytemp.extend([0, 0, 1])

            #answer to questions should be same as input minus controls
            xtemp = np.nan_to_num(xtemp)
            ytemp = np.nan_to_num(ytemp)
            #print(xtemp)
            #print(ytemp)
            sets.append({
                'x': xtemp,
                'y': ytemp
            })
        except:
            #print('fail')
            pass


    print('num sets from game:', len(sets))
    return sets


for replay in data:
    repdat = replay.get_pandas()
    repproto = replay.get_proto()
    if repproto.game_metadata.team_size != 3:
        print('nontrip')
    else:
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

        #controls = get_controls(game)
        #print(controls.iloc[200:250])
        #print(list(controls))
        #print(controltoinput(controls, 0))
        #print(createsets(game))

        #print(game['frames'], game['ball'].shape)
        #print('test')


        gamecount += 1
        datasets.extend(createsets(game))


write_name = 'set1_' + str(gamecount) + '.pset'
pickle.dump(datasets, open(write_name, 'wb'), -1)

print('done')