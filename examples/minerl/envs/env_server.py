import socket
import sys
import traceback
import gym
import logging
import minerl
import struct
import argparse
import os
import tempfile
import fcntl
import getpass

from threading import Thread


try:
    import cPickle as pickle
except ImportError:
    import pickle


class EnvServer:
    def __init__(self, handler, host: str = "127.0.0.1", port: int = 9999):
        self.handler = handler
        self.host = host
        self.port = port
        self.soc = None

    def start_server(self):
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # SO_REUSEADDR flag tells the kernel to reuse a local socket in TIME_WAIT state,
        # without waiting for its natural timeout to expire
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        logging.info("Socket created")

        try:
            soc.bind((self.host, self.port))
        except:
            logging.error("Bind failed. Error : " + str(sys.exc_info()))
            sys.exit()

        soc.listen(5)  # queue up to 5 requests
        logging.info("Socket now listening")
        self.soc = soc

    def listen(self):
        # infinite loop- do not reset for every requests
        try:
            while True:
                connection, address = self.soc.accept()
                ip, port = str(address[0]), str(address[1])
                logging.info("Connected with " + ip + ":" + port)

                try:
                    Thread(target=self.client_thread, args=(connection,)).start()
                except:
                    logging.error("Thread did not start.")
                    traceback.print_exc()
        finally:
            self.soc.close()

    def client_thread(self, connection):
        is_active = True
        client_address = connection.getpeername()
        try:
            while is_active:
                client_input = receive_data(connection)

                if type(client_input) is dict:
                    handler = client_input["type"]
                    if handler == "close":
                        send_data(connection, self.handler.message_dict[handler](client_address, client_input["data"]))
                        connection.close()
                        is_active = False
                    elif handler in self.handler.message_dict:
                        send_data(connection, self.handler.message_dict[handler](client_address, client_input["data"]))
                    else:
                        send_data(connection, Exception("invalid type"))
                        logging.error(Exception("Invalid type"))
                else:
                    try:
                        send_data(connection, Exception("invalid message"))
                    except Exception:
                        break
        finally:
            self.handler.release_env(self.handler.env_register, client_address)


def receive_data(connection):
    raw_msglen = recvall(connection, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    message = recvall(connection, msglen)
    decoded_input = pickle.loads(message, encoding="bytes")
    return decoded_input


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def send_data(connection, data):
    message = pickle.dumps(data, protocol=4)
    message = struct.pack('>I', len(message)) + message
    connection.sendall(message)


def gym_sync_create(env_string, thread_id):
    lock_dir = os.path.join(tempfile.gettempdir(), getpass.getuser())
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir)
    with open(os.path.join(lock_dir, "minecraft-{}.lock".format(thread_id)), "wb") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            env = gym.make(env_string)
            return env
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class Handler:
    def __init__(self, env_name: str = "MineRLTreechop-v0", num_envs: int = 1):
        self.message_dict = {
            "make": self.handle_make,
            "step": self.handle_step,
            "reset": self.handle_reset,
            "close": self.handle_close,
            "version": self.handle_version
        }
        self.env_name = env_name
        self.num_envs = num_envs
        self.env_pool = None
        self.env_register = None

    def handle_version(self, client, data):
        return self.env_name

    def handle_make(self, client, data):
        logging.info("Make %s" % data)
        if data == self.env_name:
            try:
                self.reserve_env(self.env_pool, self.env_register, client)
                logging.info("Success")
                return True
            except:
                logging.error("Failed - Exception in reserve_env")
                return False
        else:
            logging.error("Failed - Environment ID mismatch: {}".format(self.env_name))
            return False

    def handle_close(self, client, data):
        logging.info("Close")
        self.release_env(self.env_register, client)
        return True

    def handle_step(self, client, data):
        env = self.env_pool[self.env_register[client]]
        try:
            return env.step(data)
        except Exception as e:
            try:
                logging.error("EXCEPTION DURING env.step, resetting...\n{}".format(e))
                env.reset()
                return self.handle_step(client, data)
            except Exception as e:
                # assume broken env
                logging.error("EXCEPTION DURING env.step.reset, restarting_env...\n{}".format(e))
                self.restart_env(self.env_register[client])
                return self.handle_step(client, data)

    def handle_reset(self, client, data):
        logging.info("Reset: %s, %s" % (client, data))
        env = self.env_pool[self.env_register[client]]
        seed = data["seed"] if "seed" in data else None
        if seed:
            env.seed(data["seed"])
        try:
            return env.reset()
        except Exception as e:
            # assume broken env
            logging.error("EXCEPTION DURING env.reset, restarting_env...\n{}".format(e))
            return self.restart_env(self.env_register[client], seed=seed)

    @staticmethod
    def _make_env(env_name):
        logging.info("Initializing %s" % env_name)
        env = gym_sync_create(env_name)
        env.reset()
        return env

    def restart_env(self, env_id, seed=None):
        try:
            env = self.env_pool[env_id]
            env.close()
        except:
            pass

        env = gym.make(self.env_name)
        self.env_pool[env_id] = env
        if seed is not None:
            env.seed(seed)
        return env.reset()

    def startup_pool(self):
        # startup env pool

        logging.info("Starting up environment pool (%d): %s" % (self.num_envs, self.env_name))
        # p = Pool(self.num_envs)
        # envs = p.map(self._make_env, [self.env_name for i in range(self.num_envs)])
        # env_pool = {i: envs[i] for i in range(self.num_envs)}
        env_pool = {i: gym.make(self.env_name) for i in range(self.num_envs)}
        env_register = {}
        for env_id, env in env_pool.items():
            logging.info("Resetting env...")
            env.reset()
            logging.info("Done")
        logging.info("Ready!")
        self.env_pool = env_pool
        self.env_register = env_register

    def reserve_env(self, pool, register, address):
        self.release_env(register, address)
        for env_id, env in pool.items():
            if not env_id in register.values():
                register[address] = env_id
                logging.info("Reserved environment %d for %s" % (env_id, address))
                return
        raise Exception("Out of Environments!")

    @staticmethod
    def release_env(register, address):
        if address in register:
            env_id = register.pop(address)
            logging.info("Released env %d used by %s" % (env_id, address))


def parse_args():
    parser = argparse.ArgumentParser(description='Start environment server.')
    parser.add_argument("--env", choices=["MineRLTreechop-v0", "MineRLObtainDiamond-v0", "MineRLObtainDiamondDense-v0"],
                        help="Environment name")
    parser.add_argument("--port", type=int, default=9999, help="server port")
    parser.add_argument("--poolsize", type=int, default=1, help="number of environments to keep")
    return parser.parse_args()


def start(args):
    handler = Handler(env_name=args.env, num_envs=args.poolsize)
    handler.startup_pool()
    server = EnvServer(handler=handler, port=args.port)
    server.start_server()
    server.listen()


if __name__ == "__main__":
    args = parse_args()
    start(args)
