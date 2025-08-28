
import random
import numpy as np
from CybORG_plus_plus.mini_CAGE.minimal import HOSTS
from CybORG_plus_plus.mini_CAGE.test_agent import Base_agent


class Base_agent:
    def __init__(self):
        pass

    def train(self):
        pass

    def get_action(self):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self):
        pass


class B_line_minimal(Base_agent):
    """ A minimal Red agent for the CybORG mini-CAGE environment, copying the B-line agent from the original CybORG repo.

    The observation space given to the red agent is a 40-element vector, where the first element is the success flag (0
    or 1), and the next 39 elements represent the state of each host in the network.
    The state of each host is represented as a 3-element vector and are in order of the hosts variable in the init
    below:
    - [0] Scan state (-1 if unknown or not yet discovered, 0 if scanned but not exploited, 1 if exploited or more)
    - [1] Exploit state (0 if not exploited, 1 if exploited)
    - [2] Privilege escalation state (0 if not privileged red access, 1 if privileged red access)
    """
    def __init__(self):
        super().__init__()
        self.action = 0
        self.last_host = None  # index of most-recent target host
        self.last_action = None  # np.array([[id]]) sent last step
        self.jumps = [0, 1, 2, 2, 2, 2, 5, 5, 5, 5, 9, 9, 9, 12, 13]
        self.first_user_host = None

        hosts = [
            'def', 'ent0', 'ent1', 'ent2', 'ophost0',
            'ophost1', 'ophost2', 'opserv',
            'user0', 'user1', 'user2', 'user3', 'user4'
        ]
        self.hosts = {h: i for i, h in enumerate(hosts)}

    # ────────────────────  USER-SUBNET HELPERS  ───────────────────────
    def DiscoverRemoteSystems_user(self):
        return 3  # DRS user subnet

    def DiscoverNetworkServices_user(self, host, obs):
        if host not in (9, 10, 11, 12):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [0, 0, 0]).all():  # not scanned yet
            return {9: 13, 10: 14, 11: 15, 12: 16}[host]
        return None  # already scanned

    def ExploitRemoteService_user(self, host, obs):
        if host not in (9, 10, 11, 12):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        # legal pre-condition: scanned but not yet exploited
        if (row == [1, 0, 0]).all():
            return {9: 26, 10: 27, 11: 28, 12: 29}[host]

        # idempotent success: already compromised or escalated
        if (row == [1, 1, 0]).all() or (row == [1, 0, 1]).all():
            return {9: 26, 10: 27, 11: 28, 12: 29}[host]

        return None  # can’t exploit yet

    def PrivilegeEscalate_user(self, host, obs):
        if host not in (9, 10, 11, 12):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        # legal: exploited but not escalated
        if (row == [1, 1, 0]).all():
            return {9: 39, 10: 40, 11: 41, 12: 42}[host]

        # idempotent: already escalated
        if (row == [1, 0, 1]).all():
            return {9: 39, 10: 40, 11: 41, 12: 42}[host]

        return None  # can’t escalate yet

    # ───────────────── ENTERPRISE-SUBNET HELPERS (unchanged pre-checks) ─────────
    def DiscoverNetworkServices_ent(self, host, obs, last_host):
        if host not in (1, 2, 3):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        if host in (1, 2):  # ent0/ent1 gated
            ok_prev = (obs.reshape(1, -1)[0, 1:].reshape(13, 3)[last_host] == [1, 0, 1]).all()
            if not ok_prev:
                return None

        if (row == [0, 0, 0]).all() or (row == [1, 0, 0]).all():
            return {1: 5, 2: 6, 3: 7}[host]
        return None

    def ExploitRemoteService_ent(self, host, obs):
        if host not in (1, 2, 3):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        if (row == [1, 0, 0]).all():  # legal first exploit
            return {1: 18, 2: 19, 3: 20}[host]
        if (row == [1, 1, 0]).all() or (row == [1, 0, 1]).all():  # idempotent repeat
            return {1: 18, 2: 19, 3: 20}[host]
        return None

    def PrivilegeEscalate_ent(self, host, obs):
        if host not in (1, 2, 3):
            raise ValueError(f"bad host {host}")

        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]

        if (row == [1, 1, 0]).all():
            return {1: 31, 2: 32, 3: 33}[host]
        if (row == [1, 0, 1]).all():  # idempotent
            return {1: 31, 2: 32, 3: 33}[host]
        return None

    def DiscoverRemoteSystems_ent(self, host, obs):
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [1, 0, 1]).all():  # escalated host
            return 1  # DRS enterprise subnet
        return None

    # ───────────────── OP-SERVER HELPERS (same gating) ────────────────
    def DiscoverNetworkServices_op(self, host, obs):
        ok_prev = (obs.reshape(1, -1)[0, 1:].reshape(13, 3)[3] == [1, 0, 1]).all()
        if not ok_prev:
            return None
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [0, 0, 0]).all():
            return 11
        return None

    def ExploitRemoteService_op(self, host, obs):
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [1, 0, 0]).all() or row[0] == 1:
            return 24
        return None

    def PrivilegeEscalate_op(self, host, obs):
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [1, 1, 0]).all() or (row[0] == 1 and row[2] == 1):
            return 37
        return None

    def Impact(self, host, obs):
        row = obs.reshape(1, -1)[0, 1:].reshape(13, 3)[host]
        if (row == [1, 0, 1]).all():
            return 50
        return None

    # ───────────────────────────  FSM  ────────────────────────────────
    def get_action(self, observation, success=None):
        succ = observation.reshape(1, -1)[0, 0]
        if success is False:
            succ = 0

        loops = 0
        while True:
            loops += 1
            if loops > 200:
                raise RuntimeError(f"stuck: state={self.action}, host={self.last_host}")

            # advance or jump
            self.action = min(self.action + 1, 14) if succ == 1 else self.jumps[self.action]

            # 0 ─ DiscoverRemoteSystems on user subnet
            if self.action == 0:
                action_id = self.DiscoverRemoteSystems_user()

            # 1 ─ DiscoverNetworkServices on a user host
            elif self.action == 1:
                if self.first_user_host is None:  # first time this episode
                    self.first_user_host = np.random.choice([9, 10, 11, 12])
                self.last_host = self.first_user_host
                action_id = self.DiscoverNetworkServices_user(self.last_host, observation)
                if action_id is None: succ = 0; continue

            # 2 ─ Exploit user host
            elif self.action == 2:
                if self.last_host not in (9, 10, 11, 12):  # guard – re-use the saved one
                    self.last_host = self.first_user_host

                action_id = self.ExploitRemoteService_user(self.last_host, observation)
                if action_id is None: succ = 0; continue
                # idempotent repeat counts as success
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][0] == 1:
                    succ = 1

            # 3 ─ Privilege escalate user host
            elif self.action == 3:
                action_id = self.PrivilegeEscalate_user(self.last_host, observation)
                if action_id is None:
                    succ = 0
                    continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][2] == 1:
                    succ = 1

            # (enterprise, op-server, impact)  ──────────────
            elif self.action == 4:
                if self.last_host in (9, 10):
                    self.enterprise_host = 1
                else:
                    self.enterprise_host = 2
                action_id = self.DiscoverNetworkServices_ent(self.enterprise_host, observation, self.last_host)
                if action_id is None:
                    succ = 0
                    continue

            elif self.action == 5:
                self.last_host = self.enterprise_host
                action_id = self.ExploitRemoteService_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][0] == 1:
                    succ = 1


            elif self.action == 6:
                action_id = self.PrivilegeEscalate_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][2] == 1:
                    succ = 1

            elif self.action == 7:
                action_id = self.DiscoverRemoteSystems_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue

            elif self.action == 8:
                action_id = self.DiscoverNetworkServices_ent(3, observation, self.last_host)
                if action_id is None: succ = 0; continue

            elif self.action == 9:
                self.last_host = 3
                action_id = self.ExploitRemoteService_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][0] == 1:
                    succ = 1

            elif self.action == 10:
                action_id = self.PrivilegeEscalate_ent(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][2] == 1:
                    succ = 1

            elif self.action == 11:
                action_id = self.DiscoverNetworkServices_op(self.hosts['opserv'], observation)
                if action_id is None: succ = 0; continue

            elif self.action == 12:
                self.last_host = self.hosts['opserv']
                action_id = self.ExploitRemoteService_op(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][0] == 1:
                    succ = 1

            elif self.action == 13:
                action_id = self.PrivilegeEscalate_op(self.last_host, observation)
                if action_id is None: succ = 0; continue
                if observation.reshape(1, -1)[0, 1:].reshape(13, 3)[self.last_host][2] == 1:
                    succ = 1

            elif self.action == 14:
                action_id = self.Impact(self.last_host, observation)
                if action_id is None: succ = 0; continue

            self.last_action = np.array([[action_id]], dtype=np.int32)
            return self.last_action

    def reset(self):
        self.action = 0
        self.last_host = None
        self.last_action = None
        self.first_user_host = None  # first user host to scan, set on first DiscoverNetworkServices_user

