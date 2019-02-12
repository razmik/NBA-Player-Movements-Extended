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
        self.counter = 0

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
        last_default_index = 20  # len(data_frame) - 1

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

    def start(self):
        skip = 5
        image_limit = 2000
        self.show_updated(skip, image_limit)
        print('Completed.')

    def update_radius(self, i, player_circles, ball_circle, annotations, clock_info, skip, limit):

        moment = self.moments[i]
        for j, circle in enumerate(player_circles):
            try:
                circle.center = moment.players[j].x, moment.players[j].y
                annotations[j].set_position(circle.center)
            except:
                print('error', i, str(j), str(circle))
                annotations[j].set_position((0, 0))

            clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
                moment.quarter,
                int(moment.game_clock) % 3600 // 60,
                int(moment.game_clock) % 60,
                moment.shot_clock)
            clock_info.set_text(clock_test)
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF

        if i % skip == 0 and self.counter <= limit:
            plt.savefig('data/out/images/{}.jpeg'.format(self.counter), bbox_inches='tight', pad_inches=0)
            print('saved {} / {}'.format(self.counter, limit))
            self.counter += 1

        return player_circles, ball_circle

    def show_updated(self, skip=5, limit=1000):

        # Leave some space for inbound passes
        ax = plt.axes(xlim=(Constant.X_MIN,
                            Constant.X_MAX),
                      ylim=(Constant.Y_MIN,
                            Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid

        start_moment = self.moments[0]

        clock_info = ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                 color='black', horizontalalignment='center',
                                 verticalalignment='center')

        annotations = [ax.annotate(self.player_ids_dict[player.id][1], xy=[0, 0], color='w',
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for player in start_moment.players]

        player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                          for player in start_moment.players]
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
                                 color=start_moment.ball.color)
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
            fig, self.update_radius,
            fargs=(player_circles, ball_circle, annotations, clock_info, skip, limit),
            frames=len(self.moments), interval=Constant.INTERVAL)
        court = plt.imread("court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN])
        # plt.show()
        anim.save('test.mp4')
