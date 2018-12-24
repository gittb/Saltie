import numpy as np
import time

class Physicsframe():
    def __init__(self, players, ball, deltas, ballnext, mapname):
        self.players = {}
        for player in players:
            self.players[player["name"]] = player
        self.ball = ball
        self.deltas = deltas
        self.mapname = mapname
        self.ballnext = ballnext


class Gamedata():
    def __init__ (self, inarr, outarr, delta, mapname):
        self.frames = []
        for i in range(len(inarr)):
            Physicsframe(inarr[i][1], outarr[i][1]) #start here not done


class DataLoader():
    def __init__ (self, ):
        pass