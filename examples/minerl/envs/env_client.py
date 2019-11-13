import socket
import gym
import struct

import logging
from minerl.env import MineRLEnv


try:
    import cPickle as pickle
except ImportError:
    import pickle


def _send_message(socket, type, data):
    message = pickle.dumps({"type": type, "data": data}, protocol=4)
    message = struct.pack('>I', len(message)) + message
    socket.sendall(message)
    return _read_response(socket)


def _read_response(socket):
    raw_msglen = recvall(socket, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    message = recvall(socket, msglen)
    return pickle.loads(message, encoding="bytes")


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


class RemoteEnv(object):
    def __init__(self, envname, socket):
        specs = gym.envs.registration.spec(envname)
        self.action_space = specs._kwargs["action_space"]
        self.observation_space = specs._kwargs["observation_space"]
        self.action_space.noop = self.action_space.no_op
        self.socket = socket
        self.env_seed = None
        self.reward_range = MineRLEnv.reward_range
        self.metadata = MineRLEnv.metadata

    def version(self):
        return _send_message(self.socket, "version", {})

    def seed(self, seed):
        self.env_seed = seed

    def reset(self):
        if self.env_seed:
            return _send_message(self.socket, "reset", {"seed": self.env_seed})
        else:
            return _send_message(self.socket, "reset", {})

    def step(self, action):
        return _send_message(self.socket, "step", action)

    def close(self):
        _send_message(self.socket, "close", None)
        self.socket.close()


class RemoteGym(object):
    def __init__(self, host="localhost", port=9999):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        try:
            soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            soc.connect((self.host, self.port))
        except Exception as e:
            logging.error("Connection error - {}".format(e))
            raise
        self.socket = soc

    def version(self):
        response = _send_message(self.socket, "version", None)
        return response

    def make(self, env_name):
        response = _send_message(self.socket, "make", env_name)
        if not response:
            raise Exception("unable to make environment")
        else:
            return RemoteEnv(env_name, self.socket)
