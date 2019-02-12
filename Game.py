import pandas as pd
from Event import Event
from Team import Team
from Constant import Constant


class Game:
    """A class for keeping info about the games"""
    def __init__(self, path_to_json, event_index):
        # self.events = None
        self.home_team = None
        self.guest_team = None
        self.event = None
        self.path_to_json = path_to_json
        self.event_index = event_index

        self.all_events = []

    def read_json(self):
        data_frame = pd.read_json(self.path_to_json)
        last_default_index = len(data_frame) - 1
        self.event_index = min(self.event_index, last_default_index)
        index = self.event_index

        print(Constant.MESSAGE + str(last_default_index))
        event = data_frame['events'][index]
        self.event = Event(event)
        self.home_team = Team(event['home']['teamid'])
        self.guest_team = Team(event['visitor']['teamid'])

        # Set up all data
        self.all_events = [Event(e) for e in data_frame['events']]

    def start(self):
        self.event.show_updated()

    def start_all(self):

        for count, event in enumerate(self.all_events):
            self.event.show()


