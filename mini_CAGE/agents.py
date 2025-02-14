
import numpy as np

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

class Blue_sleep(Base_agent):
    '''
    Inactive agent compatible with CAGE implementations.
    This agent always selects the "sleep" action (action 0).
    '''
    def __init__(self, *args, **kwargs):
        super(Base_agent).__init__(*args, **kwargs)

    def get_action(self, observation, *args, **kwargs):
        """
        Always returns action 0 (sleep) for every environment instance.
        """
        return np.zeros((observation.shape[0], 1)).astype(int)

    def end_episode(self):
        """
        No special end-of-episode behavior needed.
        """
        pass

class B_line_minimal(Base_agent):
    '''
    B_line implementation that is compatible 
    with both minimal and default CAGE.
    '''
    def __init__(self, *args, **kwargs):
        super(Base_agent).__init__(*args, **kwargs)

    def get_action(self, observation, *args, **kwargs):
    
        # convert to 2D
        # decompose the state
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        host_info = observation[:, 1:].reshape(observation.shape[0], -1, 3)
        actions = np.zeros(observation.shape[0])
        action_selected = np.zeros_like(actions)
        
        # scan subnet 3
        actions, action_selected = self._scan_subnet(
            subnet_idx=2, host_idx=12, host_info=host_info,
             actions=actions, action_selected=action_selected)

        # select action for user 1
        actions, action_selected = self._check_host(
            host_index=9, action_selected=action_selected, 
            actions=actions, host_info=host_info)

        # select action for ent 1
        actions, action_selected = self._check_host(
            host_index=2, action_selected=action_selected, 
            actions=actions, host_info=host_info)

        # scan subnet 1
        actions, action_selected = self._scan_subnet(
            subnet_idx=0, host_idx=0, host_info=host_info,
             actions=actions, action_selected=action_selected)

        # select action for ent 2
        actions, action_selected = self._check_host(
            host_index=3, action_selected=action_selected, 
            actions=actions, host_info=host_info)

        # select action for opserver
        actions, action_selected = self._check_host(
            host_index=7, action_selected=action_selected, 
            actions=actions, host_info=host_info)

        # impact the operational server
        at_op_server = np.invert(action_selected.astype(bool))
        if np.any(at_op_server):
            actions[at_op_server] = 50
            action_selected[at_op_server] = 1

        return actions.reshape(-1, 1).astype(int)


    def _check_host(self, host_index, action_selected, actions, host_info):
        '''Given the index of a host select to scan, exploit or escalate. '''
        
        # scan the host
        num_hosts = host_info.shape[1]
        host_scanned = host_info[:, host_index, 0] == 1
        scan_host = np.logical_and(
            np.invert(host_scanned), np.invert(action_selected.astype(bool)))
        if np.any(scan_host):
            actions[scan_host] = host_index+4
            action_selected[scan_host] = 1

        # exploit user1
        host_exploited = host_info[:, host_index, 1] == 1
        host_privileged = host_info[:, host_index, -1] == 1
        host_access = np.logical_or(host_exploited, host_privileged)
        host_exp = np.logical_and(host_scanned, np.invert(host_access))
        exp_host = np.logical_and(
            host_exp, np.invert(action_selected.astype(bool)))
        if np.any(exp_host):
            actions[exp_host] = num_hosts+4+host_index
            action_selected[exp_host] = 1

        # priv access user 1
        host_priv = np.logical_and(host_exploited, np.invert(host_privileged))
        priv_host = np.logical_and(
            host_priv, np.invert(action_selected.astype(bool)))
        if np.any(host_priv):
            actions[priv_host] = num_hosts*2+4+host_index
            action_selected[priv_host] = 1

        return actions, action_selected

    def _scan_subnet(
        self, subnet_idx, host_idx, host_info, actions, action_selected):

        subnet_unknown = host_info[:, host_idx, -1] == -1
        subnet_unknown = np.logical_and(
            subnet_unknown, np.invert(action_selected.astype(bool)))
        if np.any(subnet_unknown):
            actions[subnet_unknown] = subnet_idx+1
            action_selected[subnet_unknown] = 1
        
        return actions, action_selected

class Meander_minimal(Base_agent):
    def __init__(self, *args, **kwargs):
        super(Base_agent).__init__(*args, **kwargs)
        self.subnet_structure = np.array([
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        self.num_hosts = len(self.subnet_structure)

    def get_action(self, observation, *args, **kwargs):

        # convert to 2D
        # decompose the state
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        host_info = observation[:, 1:].reshape(observation.shape[0], -1, 3)
        actions = np.zeros(observation.shape[0])
        action_selected = np.zeros_like(actions).astype(bool)

        # impact opserver ---------------------------------------

        opserver_priv = (host_info[:, 7, -1] == 1)
        if np.any(opserver_priv):
            actions[opserver_priv] = 50
            action_selected[opserver_priv] = 1

        # subnet scan --------------------------------------------

        host_priv = (host_info[:, :, -1] == 1)[..., None]
        host_unknown = (host_info[:, :, 0] == -1)[..., None]

        # get unique subnet configuration
        unique_subnets = np.unique(self.subnet_structure)
        subnet_idx = (self.subnet_structure[:, None] == unique_subnets).astype(int)
        subnet_idx = subnet_idx[None]

        # identifty subnets with privilege and unknown hosts
        subnet_unknown = np.any(subnet_idx*host_unknown, axis=1)
        subnet_priv = np.any(subnet_idx*host_priv, axis=1)
        subnet_scan = np.logical_and(subnet_unknown, subnet_priv)
        network_scan = np.any(subnet_scan, axis=-1)
        network_scan = np.logical_and(network_scan, np.invert(action_selected))
        if np.any(network_scan):
            selected_subnet = np.argmax(subnet_scan[network_scan], axis=-1)
            actions[network_scan] = selected_subnet+1
            action_selected[network_scan] = 1

        # network scan -------------------------------------------

        host_unscanned = (host_info[:, :, 0] == 0)
        network_unscanned = np.any(host_unscanned, axis=-1)
        network_unscanned = np.logical_and(
            network_unscanned, np.invert(action_selected))
        if np.any(network_unscanned):
            row_probs = (host_unscanned[network_unscanned] != 0)/(
                np.sum((host_unscanned[network_unscanned]  != 0), axis=-1)).reshape(-1, 1)
            selected_host = (
                np.random.rand(len(row_probs), 1) < row_probs.cumsum(axis=-1)).argmax(axis=-1)
            actions[network_unscanned] = selected_host+4
            action_selected[network_unscanned] = 1
        
        # escalate exploited network -------------------------------

        host_exploited = (host_info[:, :, 1] == 1)
        network_exploits = np.any(host_exploited, axis=-1)
        network_exploits = np.logical_and(
            network_exploits, np.invert(action_selected))
        if np.any(network_exploits):
            selected_host = np.argmax(host_exploited[network_exploits], axis=-1)
            actions[network_exploits] = selected_host+2*self.num_hosts+4
            action_selected[network_exploits] = 1

        # exploit host ---------------------------------------------

        # ensure you always ignore the defender with exploitation
        host_scanned = (host_info[:, :, 0] == 1)
        host_priv = (host_info[:, :, -1] == 1)
        host_scanned[:, 0] = False
        host_exploitable = np.logical_and(host_scanned, np.invert(host_priv))
        network_exploitable = np.any(host_exploitable, axis=-1)
        network_exploitable = np.logical_and(
            network_exploitable, np.invert(action_selected))
        if np.any(network_exploitable):
            row_probs = (host_exploitable[network_exploitable] != 0)/(
                np.sum((host_exploitable[network_exploitable]  != 0), axis=-1)).reshape(-1, 1)
            selected_host = (
                np.random.rand(len(row_probs), 1) < row_probs.cumsum(axis=-1)).argmax(axis=-1)
            actions[network_exploitable] = selected_host+self.num_hosts+4
            action_selected[network_exploitable] = 1

        return actions.reshape(-1, 1)
        

class React_restore_minimal(Base_agent):
    '''
    React-restore agent compatible with minimal and default simulator.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self, *args, **kwargs):
        self.num_hosts = 13
        self.host_list = None
    
    def get_action(self, observation, *args, **kwargs):
        
        # reformat observation
        batch_size = observation.shape[0]
        if self.host_list is None:
            self.host_list = np.zeros((batch_size, self.num_hosts))
        host_info = observation[:, :4*self.num_hosts].reshape(-1, self.num_hosts, 4)

        # update the host list
        exploited_host = host_info[:, :, -1] == 1
        host_idxs = np.where(exploited_host == 1)
        if len(host_idxs[0]) > 0:
            self.host_list[host_idxs[0], host_idxs[1]] = 1

        # restore a host to default
        actions = np.zeros(batch_size)
        host_to_restore = np.any(self.host_list == 1, axis=-1)
        if np.any(host_to_restore):
            selected_host = np.argmax(self.host_list[host_to_restore], axis=-1)
            actions[host_to_restore] = selected_host+40
            filtered_hosts = np.arange(len(observation))[host_to_restore] 
            self.host_list[filtered_hosts, selected_host] = 0

        return actions.reshape(-1, 1).astype(int)
        

class Blue_sleep(Base_agent):
    '''
    Inactive agent compatible with CAGE implementations.
    '''
    def __init__(self, *args, **kwargs):
        super(Base_agent).__init__(*args, **kwargs)

    def get_action(self, observation, *args, **kwargs):
        return np.zeros((observation.shape[0], 1)).astype(int)
    
    def end_episode(self):
        pass


class Restore_decoys(React_restore_minimal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self, *args, **kwargs):
        self.decoy_order = np.array([3, 7, 2, 1, 7, 7, 1, 7, 1, 1])
        self.obs_set = None
        super().reset(*args, **kwargs)

    def get_action(self, observation, *args, **kwargs):

        # get the order of decoys
        if self.obs_set is None:
            self.obs_set = observation.shape[0]
            self.decoy_order = np.tile(
                self.decoy_order.reshape(1, -1), (self.obs_set, 1))
                
        # select the action using react restore logic
        action = super().get_action(observation, *args, **kwargs)

        # if action is zero add an appropriate decoy
        # interested in decoys on opserver, ent2, ent1, ent0
        is_sleep = (action == 0).reshape(-1)
        has_decoys = np.sum(self.decoy_order != -1, axis=-1).reshape(-1)
        valid = np.logical_and(is_sleep, has_decoys)
        if np.any(valid):
            new_action_idx = np.argmax(self.decoy_order != -1, axis=-1)
            new_action = self.decoy_order[np.arange(len(self.decoy_order)), new_action_idx] 
            self.decoy_order[np.arange(len(self.decoy_order))[is_sleep], new_action_idx[is_sleep]] = -1             
            action[is_sleep] = new_action.reshape(-1, 1)[is_sleep]+14

        return action.reshape(-1, 1).astype(int)

    
    
