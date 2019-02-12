from Team import Team

"""
The five basketball positions normally employed by organized basketball teams are;
 the point guard (PG), 
 the shooting guard (SG), 
 the small forward (SF), 
 the power forward (PF), 
 and the center (C). 
 Typically the point guard is the leader of the team on the court.
"""

class Player:
    """A class for keeping info about the players"""
    def __init__(self, player, player_dict):
        self.team = Team(player[0])
        self.id = player[1]
        self.x = player[2]
        self.y = player[3]
        self.color = self.team.color
        self.position = player_dict[self.id][2]
