import numpy as np
import time

class Physicsframe():
    def __init__(self, players, ball, deltas, next, mapname):
        self.players = {}
        for player in players:
            self.players[player["name"]] = player
        self.ball = ball
        self.deltas = deltas
        self.mapname = mapname
        self.playernext = {}
        for player in next:
            self.players[next["name"]] = player
        self.ballnext = next['ball']

class Gamedata():
    def __init__ (self, gamearr):
        self.frames = []
        for i in range(len(gamearr)):
            self.frames.append(Physicsframe(gamearr[i][0], gamearr[i][1], gamearr[i][2], gamearr[i][3]))

class DataLoader():
    def __init__ (self, data):
        self.games = []
        for game in data:
            self.games.append(Gamedata(game))