import numpy as np
import pandas as pd
from Event import Event
from Team import Team
from Constant import Constant
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib import animation
import sys


class Game:
    """A class for keeping info about the games"""
    def __init__(self, path_to_json, event_index):
        # self.events = None
        self.home_team = None
        self.guest_team = None
        self.events = []
        self.moments = []
        self.frames = None
        self.path_to_json = path_to_json
        self.event_index = event_index
        self.player_ids_dict = None

    def set_params(self, event):
        home_players = event['home']['players']
        guest_players = event['visitor']['players']
        players = home_players + guest_players
        player_ids = [player['playerid'] for player in players]
        player_names = [" ".join([player['firstname'],
                                  player['lastname']]) for player in players]
        player_jerseys = [player['jersey'] for player in players]
        player_positions = [player['position'] for player in players]
        values = list(zip(player_names, player_jerseys, player_positions))
        self.player_ids_dict = dict(zip(player_ids, values))

    def read_json(self):
        data_frame = pd.read_json(self.path_to_json)
        last_default_index = 20 # len(data_frame) - 1

        print(Constant.MESSAGE + str(last_default_index))

        for event_id in tqdm(range(0, last_default_index), desc='Processing input data'):
            event = data_frame['events'][event_id]

            if len(self.events) < 1:
                self.home_team = Team(event['home']['teamid'])
                self.guest_team = Team(event['visitor']['teamid'])
                self.set_params(event)

            curr_event = Event(event)
            self.events.append(curr_event)
            self.moments.extend(curr_event.moments)

        # Sort moments based on game clock
        # self.moments.sort(key=lambda x: x.game_clock, reverse=True)

    def start(self):
        skip = 5

        feature_list = self.create_feature_vector(skip=skip)
        # self.save_to_npy('data/out/moment_features_{}.npy'.format(skip), feature_list)
        # self.save_to_csv('data/out/moment_features_{}.csv'.format(skip), feature_list)

        # Display frames
        self.setup_frames(skip=skip)
        self.show_frames(start=10, steps=50)
        # self.save_to_npy('data/out/moment_frames_{}.npy'.format(skip), self.frames)
        print('Completed.')

    def save_to_csv(self, out_name, out_data):
        pd.DataFrame(np.array(out_data)).to_csv(out_name, index=None)
        print('Successfully created csv.', out_name)

    def save_to_npy(self, out_name, out_data):
        np.save(out_name, np.array(out_data))
        print('Successfully saved', out_name)

    def create_feature_vector(self, skip):

        def setup_single_pos(pos_arr, data, max_val):
            for p in data:
                pos_arr.extend(p)

            for i in range(max_val - len(data)):
                pos_arr.extend([0, 0, 0])

            return pos_arr

        # Setup max values for each position
        max_G, max_FG, max_F, max_GF, max_C, max_CF = 3, 1, 2, 1, 1, 1
        features_list = []

        # For each moment create a single feature vector
        for index, moment in tqdm(enumerate(self.moments), desc='Constructing Features'):

            tor_G, tor_FG, tor_F, tor_GF, tor_C, tor_CF = [], [], [], [], [], []
            other_G, other_FG, other_F, other_GF, other_C, other_CF = [], [], [], [], [], []

            if index % skip != 0:
                continue

            if len(features_list) == 2000:
                print('2000 reached.')

            # Construct an array with x,y, speed, position for each player
            position_player_tor = {'G': [], 'F-G': [], 'F': [], 'G-F': [], 'C': [], 'C-F': []}
            position_player_other = {'G': [], 'F-G': [], 'F': [], 'G-F': [], 'C': [], 'C-F': []}
            for player in moment.players:

                # calculate player speed
                speed = 0
                if index > 0:
                    player_prev_list = [p for p in self.moments[index-1].players if p.id == player.id]
                    if len(player_prev_list) > 0:
                        player_prev = player_prev_list[0]
                        speed = np.sqrt((player.x - player_prev.x) ** 2 + (player.y - player_prev.y) ** 2)

                # Add player to the list
                if player.team.name == 'TOR':
                    position_player_tor[player.position].append([min(player.x, Constant.X_MAX), min(player.y, Constant.Y_MAX), speed])
                else:
                    position_player_other[player.position].append([min(player.x, Constant.X_MAX), min(player.y, Constant.Y_MAX), speed])

            # Set tor_G, other_FG, other_F, other_GF, other_C, other_CF
            tor_G = setup_single_pos(tor_G, position_player_tor['G'], max_G)
            tor_FG = setup_single_pos(tor_FG, position_player_tor['F-G'], max_FG)
            tor_F = setup_single_pos(tor_F, position_player_tor['F'], max_F)
            tor_GF = setup_single_pos(tor_GF, position_player_tor['G-F'], max_GF)
            tor_C = setup_single_pos(tor_C, position_player_tor['C'], max_C)
            tor_CF = setup_single_pos(tor_CF, position_player_tor['C-F'], max_CF)

            other_G = setup_single_pos(other_G, position_player_other['G'], max_G)
            other_FG = setup_single_pos(other_FG, position_player_other['F-G'], max_FG)
            other_F = setup_single_pos(other_F, position_player_other['F'], max_F)
            other_GF = setup_single_pos(other_GF, position_player_other['G-F'], max_GF)
            other_C = setup_single_pos(other_C, position_player_other['C'], max_C)
            other_CF = setup_single_pos(other_CF, position_player_other['C-F'], max_CF)

            if index > 0:
                ball_prev = self.moments[index - 1].ball
                speed = np.sqrt((moment.ball.x - ball_prev.x) ** 2 + (moment.ball.y - ball_prev.y) ** 2)
            else:
                speed = 0

            ball = [moment.ball.x, moment.ball.y, speed]

            feature_vector = tor_G + tor_FG + tor_F + tor_GF + tor_C + tor_CF + other_G + other_FG + other_F + other_GF + other_C + other_CF + ball
            features_list.append(feature_vector)

        return np.asarray(features_list)

    def evaluate_positions(self, skip=1):

        overall_positions_tor = {'G': [], 'F-G': [], 'F': [], 'G-F': [], 'C': [], 'C-F': []}
        overall_positions_other = {'G': [], 'F-G': [], 'F': [], 'G-F': [], 'C': [], 'C-F': []}

        for index, moment in tqdm(enumerate(self.moments), desc='Constructing Frames'):

            current_match_pos_tor = {'G': 0, 'F-G': 0, 'F': 0, 'G-F': 0, 'C': 0, 'C-F': 0}
            current_match_pos_other = {'G': 0, 'F-G': 0, 'F': 0, 'G-F': 0, 'C': 0, 'C-F': 0}

            if index % skip != 0:
                continue

            for player in moment.players:
                if player.team.name == 'TOR':
                    current_match_pos_tor[player.position] += 1
                else:
                    current_match_pos_other[player.position] += 1

            for key in overall_positions_tor.keys():
                overall_positions_tor[key].append(current_match_pos_tor[key])
                overall_positions_other[key].append(current_match_pos_other[key])

        for key, value in overall_positions_tor.items():
            print('tor', key, sorted(list(set(value))))
        for key, value in overall_positions_other.items():
            print('other', key, sorted(list(set(value))))

    def setup_frames(self, skip=1):

        self.frames = []

        for index, moment in tqdm(enumerate(self.moments), desc='Constructing Frames'):

            if index % skip != 0:
                continue

            field = np.zeros((Constant.Y_MAX, Constant.X_MAX))
            bx, by = min(int(moment.ball.x), Constant.X_MAX-1), min(int(moment.ball.y), Constant.Y_MAX-1)
            ball_owner, min_ball_dist = None, sys.maxsize

            for player in moment. players:

                px, py = min(int(player.x), Constant.X_MAX-1), min(int(player.y), Constant.Y_MAX-1)
                field[py][px] += 50 if player.team.name == 'TOR' else -50

                p_to_ball = (px-bx) ** 2 + (py-by) ** 2
                if p_to_ball < min_ball_dist:
                    min_ball_dist = p_to_ball
                    ball_owner = (player.team.name, px, py)

            field[ball_owner[2]][ball_owner[1]] += 100 if ball_owner[0] == 'TOR' else -100

            self.frames.append(field)

    def show_frames(self, start=0, steps=-1):

        a = []
        for f in self.frames:
            a.append(np.unique(f))
        a

        end = (start+steps+1) if steps > 0 else len(self.frames)

        fig = plt.gcf()

        court = plt.imread("court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN])

        im = None
        for i in range(start, end, 1):
            if not im:
                # for the first frame generate the plot...
                plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN], cmap='jet', alpha=0.5)
                im = plt.imshow(self.frames[i])
            else:
                # ... for subsequent times only update the data
                plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN], cmap='jet', alpha=0.5)
                im.set_data(self.frames[i])
            plt.draw()
            plt.pause(0.00001)

        plt.show()

    def show_game(self):

        def update_radius(i, player_circles, ball_circle, annotations, clock_info):
            moment = self.moments[i]
            for j, circle in enumerate(player_circles):
                circle.center = moment.players[j].x, moment.players[j].y
                annotations[j].set_position(circle.center)
                clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
                    moment.quarter,
                    int(moment.game_clock) % 3600 // 60,
                    int(moment.game_clock) % 60,
                    moment.shot_clock)
                clock_info.set_text(clock_test)
            ball_circle.center = moment.ball.x, moment.ball.y
            ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
            return player_circles, ball_circle

        # Leave some space for inbound passes
        ax = plt.axes(xlim=(Constant.X_MIN,
                            Constant.X_MAX),
                      ylim=(Constant.Y_MIN,
                            Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid
        start_moment = self.moments[0]
        player_dict = self.player_ids_dict

        clock_info = ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                 color='black', horizontalalignment='center',
                                 verticalalignment='center')

        annotations = [ax.annotate(self.player_ids_dict[player.id][1], xy=[0, 0], color='w',
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for player in start_moment.players]

        # Prepare table
        sorted_players = sorted(start_moment.players, key=lambda player: player.team.id)

        home_player = sorted_players[0]
        guest_player = sorted_players[5]
        column_labels = tuple([home_player.team.name, guest_player.team.name])
        column_colours = tuple([home_player.team.color, guest_player.team.color])
        cell_colours = [column_colours for _ in range(5)]

        home_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in
                        sorted_players[:5]]
        guest_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in
                         sorted_players[5:]]
        players_data = list(zip(home_players, guest_players))

        table = plt.table(cellText=players_data,
                          colLabels=column_labels,
                          colColours=column_colours,
                          colWidths=[Constant.COL_WIDTH, Constant.COL_WIDTH],
                          loc='bottom',
                          cellColours=cell_colours,
                          fontsize=Constant.FONTSIZE,
                          cellLoc='center')
        table.scale(1, Constant.SCALE)
        table_cells = table.properties()['child_artists']
        for cell in table_cells:
            cell._text.set_color('white')

        player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                          for player in start_moment.players]
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
                                 color=start_moment.ball.color)
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
            fig, update_radius,
            fargs=(player_circles, ball_circle, annotations, clock_info),
            frames=len(self.moments), interval=Constant.INTERVAL)
        court = plt.imread("court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN])
        plt.show()
