import numpy as np

#############################################################
# TODO: need to make compatible with different architectures
#############################################################

# these are specific to the default CAGE 2 Environment -----------------------------------------------------

# ------------------------------------------------------------------------------------

# attacker and defender actions
RED_ACTIONS = ['sleep', 'remote', 'network', 'exploit', 'escalate', 'impact']
BLUE_ACTIONS = ['sleep', 'analyse', 'decoy', 'remove', 'restore']

# specify network configuration
NUM_SUBNETS = 3
HOSTS = ['def', 'ent0', 'ent1', 'ent2', 'ophost0',
         'ophost1', 'ophost2', 'opserv', 'user0', 'user1', 'user2', 'user3', 'user4']

# which hosts are connected
CONNECTED_HOSTS = [
    None, None, None, 'opserv', None, None,
    None, None, None, 'ent0', 'ent0', 'ent1', 'ent1']

# what services are already running on the machines
# -> highlights the exploit they correspond to
HOST_EXPLOITS = [

    # ent
    ['Brute'],
    ['Brute'],
    ['Brute', 'Eternal', 'Keep', 'HTTPRFI', 'HTTPSRFI'],
    ['Brute', 'Eternal', 'Keep', 'HTTPRFI', 'HTTPSRFI'],

    # op
    ['Brute'],
    ['Brute'],
    ['Brute'],
    ['Brute'],

    ##############################################
    # BUG: user3 is meant to possess SQL exploit, 
    # but is in fact replaced by bluekeep
    ##############################################

    # user
    [],
    ['Brute', 'FTP'],
    ['Eternal', 'Keep'],
    ['Keep', 'HTTPSRFI', 'HTTPRFI', 'Haraka'],
    ['Keep', 'HTTPSRFI', 'HTTPRFI', 'Haraka', 'SQL']
]

# only some user exploits are rewarded
# this highlights the exploits which are
REWARDED_EXPLOITS = [

    # ent
    [],
    [],
    ['Keep', 'Eternal'],
    ['Keep', 'Eternal'],

    # op
    [],
    [],
    [],
    [],

    # user
    [],
    ['FTP'],
    ['Eternal'],
    ['Keep', 'Haraka'],
    ['SQL', 'Haraka']

]

# what decoy options are available for each host
# ordered based on the Cardiff implementation
HOST_DECOYS = [

    # ent
    # ['Haraka', 'Tomcat', 'Apache', 'Vsftpd'],
    [],
    ['Haraka', 'Tomcat', 'Vsftpd', 'Apache'],
    ['Femitter'],
    ['Femitter'],

    # op
    [],
    [],
    [],
    ['Haraka', 'Apache', 'Tomcat', 'Vsftpd'],

    # user
    [],
    ['Apache', 'Tomcat', 'SMSS', 'Svchost'],
    ['Femitter', 'Tomcat', 'Apache', 'SSHD'],
    ['Vsftpd', 'SSHD'],
    ['Vsftpd']
]

# list all the decoy and exploit options
# ranked in order of priority (high to low)
EXPLOITS = ['FTP', 'Haraka', 'SQL', 'HTTPSRFI', 'HTTPRFI', 'Eternal', 'Keep', 'Brute']
DECOYS = ['Femitter', 'Vsftpd', 'Apache', 'Haraka', 'SSHD', 'SMSS', 'Tomcat', 'Svchost']


def exploits_to_decoys(remove_bugs):
    '''Give an exploit index and return the compatible decoys.'''

    ######################################################
    # BUG: vsftp has the wrong port attached to it
    # -> so in fact it actually stops HTTPRFI
    # -> it basically deploying an apache server instead
    ######################################################
    ftp_decoys = [0]
    sql_decoys = [2, 6, 1]
    httprfi_decoys = [2, 1]
    if remove_bugs:
        ftp_decoys = [0, 1]
        sql_decoys = [2, 6]
        httprfi_decoys = [2]

    # maps the exploit to the decoys that can stop it
    mapping = np.zeros((len(EXPLOITS), len(DECOYS)))
    mapping[0, ftp_decoys] = 1  # FTP       ->  Femitter (-Vsftp)
    mapping[1, [3]] = 1  # Haraka    ->  Haraka
    mapping[2, sql_decoys] = 1  # SQL       ->  Apache, Tomcat (+Vsftp)
    mapping[3, [6]] = 1  # HTTPSRFI  ->  Tomcat
    mapping[4, httprfi_decoys] = 1  # HTTPRFI   ->  Apache (+ Vsftp)
    mapping[5, [5]] = 1  # Eternal   ->  SMSS
    mapping[6, [7]] = 1  # Keep      ->  Svchost
    mapping[7, [4]] = 1  # Brute     ->  SSHD
    return mapping


def construct_exploit_rew():
    ''''''
    mapping = np.zeros((len(HOSTS), len(EXPLOITS)))
    for idx, hosts in enumerate(REWARDED_EXPLOITS):
        for h in hosts:
            h_idx = EXPLOITS.index(h)
            mapping[idx, h_idx] = 1
    return mapping


def create_subnets(num_nodes=13):
    '''
    Divide the nodes into subnets.
    '''
    subnets = np.zeros(num_nodes)
    subnets[4:8] = 1
    subnets[8:] = 2

    return subnets


def get_host_priority(hosts):
    '''
    Designate the hosts for reward classification.
    '''
    hosts = np.array(hosts)
    host_priority = np.zeros_like(hosts, dtype=np.int32).reshape(-1)

    # user and op hosts
    # low priority hosts
    user_idxs = np.char.find(hosts, 'user')
    host_priority[np.nonzero(user_idxs + 1)[0]] = 1
    op_idxs = np.char.find(hosts, 'ophost')
    host_priority[np.nonzero(op_idxs + 1)[0]] = 1

    # enterprise and defender
    # medium priority
    ent_idxs = np.char.find(hosts, 'ent')
    host_priority[np.nonzero(ent_idxs + 1)[0]] = 2
    def_idxs = np.char.find(hosts, 'def')
    host_priority[np.nonzero(def_idxs + 1)[0]] = 2

    # opserver
    # lowest priority
    opserv_idxs = np.char.find(hosts, 'opserv')
    host_priority[np.nonzero(opserv_idxs + 1)[0]] = 3

    return host_priority


# --------------------------------------------------------------------------------------------

def default_host_exploits(remove_bug=False):
    '''Get the default exploits for each network host.'''
    mapping = np.zeros((len(HOST_EXPLOITS), len(EXPLOITS)))

    # add in SQL exploit on user3
    host_exploits = HOST_EXPLOITS
    if remove_bug:
        host_exploits[11].append('SQL')

    for i, host in enumerate(host_exploits):
        for _, exploit in enumerate(host):
            mapping[i][EXPLOITS.index(exploit)] = 1
    return mapping


def default_defender_decoys():
    '''Return the mapping of hosts to decoys.'''
    mapping = np.zeros((len(HOST_DECOYS), len(DECOYS)))
    for i, host in enumerate(HOST_DECOYS):
        for j, decoy in enumerate(host):
            mapping[i][DECOYS.index(decoy)] = (len(host) - j)
    return mapping


def action_mapping():
    '''
    Return the integer to string mapping for red/blue agent.
    '''

    # populate the red map
    red_a_map = [RED_ACTIONS[0]]
    for i in range(NUM_SUBNETS):
        red_a_map.append(f'{RED_ACTIONS[1]}_sub{i}')
    for a in RED_ACTIONS[2:]:
        for h in HOSTS:
            red_a_map.append(f'{a}_{h}')

    # populate the blue map
    blue_a_map = [BLUE_ACTIONS[0]]
    for a in BLUE_ACTIONS[1:]:
        for h in HOSTS:
            blue_a_map.append(f'{a}_{h}')

    return {'Blue': blue_a_map, 'Red': red_a_map}


def check_red_access(obs):
    '''
    Determine the access the of the red agent within 
    the network, including known hosts, scanned_hosts and host_access.
    '''

    # extract the host information
    batch_size = obs.shape[0]
    host_access = obs.reshape(batch_size, -1, 3)[:, :, 1:]

    # check which networks have been scanned
    # -> used to identify exploitable networks
    scanned = obs.reshape(batch_size, -1, 3)[:, :, 0] == 1

    # check subnet privlege
    # -> used to enable remote service scan
    priv_1 = np.any(host_access[:, :4, 1] == 1, axis=-1)
    priv_2 = np.any(host_access[:, 4:8, 1] == 1, axis=-1)
    priv_3 = np.any(host_access[:, 8:, 1] == 1, axis=-1)
    subnet_priv = np.concatenate([
        priv_1.reshape(-1, 1),
        priv_2.reshape(-1, 1),
        priv_3.reshape(-1, 1)], axis=-1)

    # check user and privlege access
    # -> known used for remote network scan
    # -> user used for escalate privleges
    # -> privleged used to impact
    known_hosts = host_access[:, :, 0] != -1
    user_access = np.any(host_access == 1, axis=-1)
    privleged_access = host_access[:, :, 1] == 1

    return known_hosts, scanned, user_access, privleged_access, subnet_priv


def get_possible_red_actions(
        user_access, priv_access, known_hosts, subnets, scanned):
    '''Return a list of valid red team actions.'''

    # add discover remote services actions
    # remove when all networks on host are known
    known_1 = np.all(known_hosts[:, :4], axis=-1)
    known_2 = np.all(known_hosts[:, 4:8], axis=-1)
    known_3 = np.all(known_hosts[:, 8:], axis=-1)
    known_subnets = np.concatenate([
        known_1.reshape(-1, 1),
        known_2.reshape(-1, 1),
        known_3.reshape(-1, 1)], axis=-1) * 1

    # create a mask
    # ensure that sleep action is always allowed
    batch_size = user_access.shape[0]
    action_mask = np.zeros(
        (batch_size, NUM_SUBNETS + len(HOSTS) * len(RED_ACTIONS[2:]) + 1))
    action_mask[:, 0] = 1

    # keep track of actions including those not already taken
    full_action_mask = action_mask.copy()

    # add discover remote services
    subnet_indices = np.nonzero(subnets * (1 - known_subnets))
    action_mask[subnet_indices[0], subnet_indices[1] + 1] = 1
    full_action_mask[np.nonzero(subnets)[0], np.nonzero(subnets)[1] + 1] = 1
    added_actions = NUM_SUBNETS + 1

    # add discover network services
    # remove if already scanned
    known_hosts_indices = np.nonzero(known_hosts * (1 - scanned))
    action_mask[known_hosts_indices[0], known_hosts_indices[1] + added_actions] = 1
    full_action_mask[np.nonzero(known_hosts)[0], np.nonzero(known_hosts)[1] + added_actions] = 1
    added_actions += len(HOSTS)

    # add exploit remote services
    # remove if already exploited
    scanned_indices = np.nonzero(scanned * (1 - user_access))
    action_mask[scanned_indices[0], scanned_indices[1] + added_actions] = 1
    full_action_mask[np.nonzero(scanned)[0], np.nonzero(scanned)[1] + added_actions] = 1
    added_actions += len(HOSTS)

    # escalate privileges
    # remove if already escalated
    user_access_indices = np.nonzero(user_access * (1 - priv_access))
    action_mask[user_access_indices[0], user_access_indices[1] + added_actions] = 1
    full_action_mask[np.nonzero(user_access)[0], np.nonzero(user_access)[1] + added_actions] = 1
    added_actions += len(HOSTS)

    # impact components
    priv_access_indices = np.nonzero(priv_access)
    action_mask[priv_access_indices[0], priv_access_indices[1] + added_actions] = 1
    full_action_mask[priv_access_indices[0], priv_access_indices[1] + added_actions] = 1
    added_actions += len(HOSTS)

    #############################################
    # NOTE: this is specific to this environment
    #############################################
    # defender can never exploited
    action_mask[:, 17] = 0

    return action_mask, full_action_mask

def update_red(state, action, subnet_loc, processes, impacted, femitter_placed, remove_bug=False):
    '''
    Update the environmental state following a red action.
    '''


    # copy the current state
    action_reward = np.zeros((state.shape[0], 1))
    next_state = state.copy()

    # identify actions that are possible
    known, scanned, user, priv, subnet = check_red_access(state)
    red_mask, full_red_mask = get_possible_red_actions(
        user, priv, known, subnet, scanned)

    # filter actions that are not allowed
    action_filter = red_mask[
        np.arange(len(action)), action.reshape(-1).astype(int)]
    success = np.ones((state.shape[0], 1))
    new_impacted = impacted.copy()

    # log the selected exploit process
    selected_exploit_idx = -np.ones(state.shape[0])

    is_subnet_scan = np.logical_and((action < (NUM_SUBNETS + 1)).reshape(-1), (action != 0).reshape(-1))

    if np.any(is_subnet_scan):
        valid = is_subnet_scan

        # Build subnet mask: find hosts belonging to the scanned subnet
        subnet_number = (action[valid] - 1).reshape(-1, 1)  # Subnet indices start from 0
        is_subnet = subnet_loc[valid] == subnet_number  # Mask for hosts in the targeted subnet

        # Update hosts in that subnet from -1 to 0
        mod_state = next_state[valid].copy().reshape(-1, len(HOSTS), 3)
        mod_state[is_subnet, :] = 0

        # Ensure we don't overwrite other hosts (if needed)
        mod_state = np.maximum(next_state[valid], mod_state.reshape(mod_state.shape[0], -1))
        next_state[valid] = mod_state

    # if valid actions remain
    if np.any(action_filter):

        # extract the host and action type
        # displace as first actions are all subnet related
        host_alloc = ((action - (NUM_SUBNETS + 1)) % len(HOSTS)).reshape(-1).astype(int)
        action_alloc = np.floor((action - (NUM_SUBNETS + 1)) / len(HOSTS)).reshape(-1).astype(int)
        # print(f'action alloc: {action_alloc}')

        # check if it is sleep
        is_sleep = (action == 0).reshape(-1)

        # subnet scan  ------------------------------------------------------

        # update the subnet following a scan
        # check if action is subnet action and if the action is allowed
        # create duplicate state and set subnet to 0 from -1
        # ensure this does not overwrite previously defined states
        is_remote = np.logical_and((action < (NUM_SUBNETS + 1)).reshape(-1), 1 - is_sleep)
        if np.any(is_remote):
            valid = np.logical_and(action_filter, is_remote)
            if np.any(valid):
                is_subnet = (subnet_loc[valid] == np.tile(action[valid] - 1, (1, len(HOSTS))))
                mod_state = next_state[valid].copy().reshape(-1, len(HOSTS), 3)
                mod_state[is_subnet, :] = 0
                mod_state = np.maximum(next_state[valid], mod_state.reshape(mod_state.shape[0], -1))
                next_state[valid] = mod_state

        # host scan -----------------------------------------------------

        # update the host with scanned (1, 0, 0)
        # check if action corresponds to scan and if not remote search
        # ensure this does not overwrite user/privleged access
        is_network = np.logical_and((action_alloc.reshape(-1) == 0), 1 - is_remote)
        if np.any(is_network):
            valid = np.logical_and(action_filter, is_network)
            if np.any(valid):
                mod_state = next_state[valid].copy().reshape(-1, len(HOSTS), 3)
                host = host_alloc[valid]

                mod_state[np.arange(len(host)), host] = np.array([1, 0, 0])
                mod_state = np.maximum(next_state[valid], mod_state.reshape(mod_state.shape[0], -1))
                next_state[valid] = mod_state.reshape(mod_state.shape[0], -1)

        # exploit user --------------------------------------------------

        # update the host to user access (1, 1, 0)
        # check if corresponds to exploit and is note remote search
        # don't allow exploitation of the defender
        is_user = np.logical_and((action_alloc.reshape(-1) == 1), 1 - is_remote)
        if np.any(is_user):
            valid = np.logical_and(action_filter, is_user)
            if np.any(valid):

                proc = processes[valid]
                host = host_alloc[valid]

                # extract the priority process
                host_processes = proc[
                    np.arange(len(proc)), host].reshape(-1, proc.shape[-1])

                ##################################
                # BUG: femitter previously placed
                # -> adds an trial_PPO failure mode
                #################################

                # add femitter if previously placed
                fem_on_host = femitter_placed[np.arange(len(host_alloc)), host_alloc]
                if np.any(fem_on_host[valid]) and (not remove_bug):
                    host_processes[fem_on_host[valid], 0] = -1

                priority_idx = np.argmax(host_processes != 0, axis=-1).reshape(-1)
                priority_process = host_processes[
                    np.arange(len(host_processes)),
                    priority_idx]

                # extract the non-priority process
                non_priority_processes = host_processes.copy()
                non_priority_processes[
                    np.arange(len(host_processes)),
                    np.argmax(host_processes != 0, axis=-1).reshape(-1)] = 0
                non_priority_process = priority_process.copy()

                # ensure additional processes are not extracted for a single process machine
                multi_proc = np.any((non_priority_processes != 0), axis=-1).reshape(-1)
                non_prior_idx = -np.ones(len(multi_proc))
                if np.any(multi_proc):
                    row_probs = (non_priority_processes[multi_proc] != 0) / (
                        np.sum((non_priority_processes[multi_proc] != 0), axis=-1)).reshape(-1, 1)
                    non_prior_choices = (
                            np.random.rand(len(row_probs), 1) < row_probs.cumsum(axis=-1)).argmax(axis=-1)
                    non_prior_copy = non_priority_processes[multi_proc].copy()
                    non_priority_process[multi_proc] = non_prior_copy[
                        np.arange(len(non_prior_copy)),
                        non_prior_choices.reshape(-1)]
                    non_prior_idx[multi_proc.astype(bool)] = non_prior_choices

                # calculate the success
                # ensure single process tasks are selected
                priority_mask = np.random.choice(
                    [0, 1], size=priority_process.shape, p=[0.25, 0.75])
                if not np.all(multi_proc):
                    not_multi_proc = (1 - multi_proc).astype(bool)
                    non_priority_process[not_multi_proc] = priority_process[not_multi_proc]
                    non_prior_idx[not_multi_proc] = priority_idx[not_multi_proc]
                selected_process = np.where(
                    priority_mask.astype(bool), priority_process, non_priority_process)
                selected_idx = np.where(priority_mask.astype(bool), priority_idx, non_prior_idx)
                exploit_success = np.zeros((valid.shape[0])).astype(bool)
                exploit_success[valid] = (selected_process == 1).reshape(-1)

                # ensure process blue eternal fails on ent1 and ent2
                eternal_working = np.zeros((valid.shape[0])).astype(bool)
                eternal_working[valid] = np.invert(np.logical_and(
                    (selected_idx == 5), np.logical_or(host == 2, host == 3)))
                exploit_success = np.logical_and(exploit_success, eternal_working)

                ######################################
                # BUG: bluekeep always fails on user3
                ######################################
                if (not remove_bug):
                    blue_u3 = np.zeros((valid.shape[0])).astype(bool)
                    blue_u3[valid] = np.invert(np.logical_and(
                        selected_idx == 6, host == 11))
                    exploit_success = np.logical_and(exploit_success, blue_u3)

                # add exploits
                old_valid = valid.copy()
                valid = np.logical_and(valid, exploit_success)
                if np.any(valid):
                    mod_state = next_state[valid].copy().reshape(-1, len(HOSTS), 3)
                    mod_state[np.arange(len(host_alloc[valid])), host_alloc[valid]] = np.array([1, 1, 0])
                    # Add this debug print here:
                    next_state[valid] = mod_state.reshape(mod_state.shape[0], -1)
                    selected_exploit_idx[valid] = selected_idx[valid[old_valid]]

                # triggering a decoy counts as a failure
                exploit_fail = np.logical_and(
                    old_valid, np.invert(exploit_success.astype(bool)))
                if np.any(exploit_fail):
                    success[exploit_fail] = -1

        # escalate privleges --------------------------------------------------

        # update the host to privileged (1, 0, 1)
        # also need to open further paths within the network
        is_priv = np.logical_and((action_alloc.reshape(-1) == 2), 1 - is_remote)
        if np.any(is_priv):
            valid = np.logical_and(action_filter, is_priv)
            if np.any(valid):
                host = host_alloc[valid]
                mod_state = next_state[valid].copy().reshape(-1, len(HOSTS), 3)
                mod_state[np.arange(len(host)), host] = np.array([1, 0, 1])
                next_state[valid] = mod_state.reshape(mod_state.shape[0], -1)

                # check for further network paths
                # user 1 -> ent 1 ; user 2 -> ent 1
                # user 3 -> ent 1 ; user 4 -> ent 0
                # ent  2 -> op server
                host_link = np.array(CONNECTED_HOSTS)[host]
                host_matrix = np.tile(
                    np.array(HOSTS).reshape(1, -1), (mod_state.shape[0], 1))
                host_mask = (host_matrix[:, np.newaxis] == host_link[:, np.newaxis, np.newaxis])
                host_idx = np.where(host_mask)[-1].reshape(-1)
                existing_link = np.zeros(len(valid))
                existing_link[valid] = (host_link != None)
                existing_link = existing_link.astype(bool)

                # ensure this doesn't overwrite an existing host
                is_link = np.zeros(len(valid)).astype(bool)
                link_in_mask = np.any(host_mask, axis=-1).reshape(-1)
                is_link[valid] = link_in_mask
                mod_state = next_state[is_link].copy().reshape(-1, len(HOSTS), 3)

                new_host = np.zeros(len(valid)).astype(bool)
                blank_host = mod_state[np.arange(len(host_idx)), host_idx.astype(int), 0] == -1
                new_host[is_link] = blank_host
                existing_link = np.logical_and(new_host, existing_link)
                host_idx = host_idx[blank_host]

                valid = np.logical_and(valid, existing_link)
                if np.any(valid):
                    mod_state = next_state[valid].copy().reshape(-1, len(HOSTS), 3)
                    new_host_idx = host_idx[valid[existing_link]]
                    mod_state[np.arange(len(new_host_idx)), new_host_idx.astype(int)] = np.array([0, 0, 0])
                    next_state[valid] = mod_state.reshape(mod_state.shape[0], -1)

        # impact host ------------------------------------------------

        # impact the hosts
        is_impact = np.logical_and((action_alloc.reshape(-1) == 3), 1 - is_remote)
        if np.any(is_impact):
            valid = np.logical_and(action_filter, is_impact)
            if np.any(valid):
                host = host_alloc[valid]
                impact_copy = new_impacted[valid].copy()
                impact_copy[np.arange(len(host)), host] = 1
                new_impacted[valid] = impact_copy

    return next_state, action_reward, success, new_impacted, selected_exploit_idx


def check_blue_action(observation, decoys):
    '''
    Check which blue actions are available.
    '''

    if len(observation.shape) == 1:
        observation = observation.reshape(1, -1)
    batch_size = observation.shape[0]
    action_mask = np.zeros((batch_size, len(HOSTS) * len(BLUE_ACTIONS[1:]) + 1))
    action_mask[:, 0] = 1

    # add analyse actions
    action_mask[:, range(1, len(HOSTS) + 1)] = 1
    added_actions = len(HOSTS) + 1

    # add decoy actions for those that still have decoys
    subnet_decoy_idxs = np.nonzero(decoys > 0)
    action_mask[subnet_decoy_idxs[0], subnet_decoy_idxs[1] + added_actions] = 1
    added_actions += len(HOSTS)

    # add remove actions
    action_mask[:, range(added_actions, len(HOSTS) + added_actions)] = 1
    added_actions += len(HOSTS)

    # add restore actions
    action_mask[:, range(added_actions, len(HOSTS) + added_actions)] = 1

    #############################################
    # NOTE: this is specific to this environment
    #############################################
    # user0 cannot be restored
    action_mask[:, 48] = 0

    return action_mask


def update_blue(
        state, updated_state, action, decoys, processes, proc_map, impacted, femitter_placed):
    '''
    Update the environmental state following a blue action.
    '''

    # copy the current state
    action_reward = np.zeros((state.shape[0], 1))
    next_state = updated_state.copy()
    orig_state = state.copy()

    # filter actions that are not allowed
    # user0 cannot be restored
    blue_mask = check_blue_action(updated_state, decoys)
    action_filter = blue_mask[
        np.arange(len(action)), action.reshape(-1).astype(int)]
    success = -np.ones((state.shape[0], 1))
    new_decoys = decoys.copy()
    new_processes = processes.copy()
    new_impacted = impacted.copy()
    decoy_reset = np.zeros((state.shape[0], len(HOSTS)))

    # if valid actions remain
    if np.any(action_filter):

        # extract the host and action type
        # displace as first actions are all subnet related
        host_alloc = ((action - 1) % len(HOSTS)).reshape(-1).astype(int)
        action_alloc = (np.floor((action - 1) / len(HOSTS))).reshape(-1).astype(int)

        # check if it is sleep
        is_sleep = (action == 0).reshape(-1)

        # analyse host ------------------------------------------------

        # analyse a host
        # analys has no effect on the environment
        is_analyse = np.logical_and(action_alloc.reshape(-1) == 0, 1 - is_sleep)
        if np.any(is_analyse):
            valid = np.logical_and(action_filter, is_analyse)
            if np.any(valid):
                success[valid] = 1

        # decoy placement ---------------------------------------------

        # place a decoy service
        is_decoy = np.logical_and(action_alloc.reshape(-1) == 1, 1 - is_sleep)
        if np.any(is_decoy):
            valid = np.logical_and(action_filter, is_decoy)
            if np.any(valid):

                dec = decoys[valid]
                host = host_alloc[valid]
                proc = processes[valid]

                # extract host decoys
                # remove from available decoys
                available_decoys = dec[np.arange(len(dec)), host, :]
                best_decoy = np.argmax(available_decoys, axis=-1)
                decoys_exist = np.any(available_decoys, axis=-1)
                decoys_temp = decoys[valid].copy()
                decoys_temp[np.arange(len(dec)), host, best_decoy] = 0

                # extract new process and add to current processes
                # ensure there is not already a process of this type running
                new_proc = proc_map.T[best_decoy]
                processes_temp = proc.copy()
                proc_spaces = (new_proc - processes_temp[
                    np.arange(len(proc)), host]) > 0
                proc_free = np.any(proc_spaces, axis=-1)
                selected_proc = np.argmax(proc_spaces, axis=-1).astype(int)
                processes_temp[np.arange(len(proc)), host, selected_proc] = -1

                # update the storage
                d_exist = np.zeros(valid.shape[0])
                p_free = np.zeros(valid.shape[0])
                d_exist[valid] = decoys_exist
                p_free[valid] = proc_free

                valid = np.logical_and(
                    valid, np.logical_and(d_exist, p_free)).astype(bool).reshape(-1)
                new_processes[valid] = processes_temp[np.logical_and(decoys_exist, proc_free)]
                new_decoys[valid] = decoys_temp[np.logical_and(decoys_exist, proc_free)]
                success[valid] = 1

                # if femitter is placed log this
                femitter_on_host = (best_decoy == 0)[
                    np.logical_and(decoys_exist, proc_free)]
                femitter_slice = np.zeros(valid.shape)
                femitter_slice[valid] = femitter_on_host
                femitter_slice = np.logical_and(femitter_slice, valid)
                if np.any(femitter_slice):
                    femitter_host = host_alloc[femitter_slice]
                    femitter_placed[np.arange(len(host_alloc))[femitter_slice], femitter_host] = True

        # host removal ------------------------------------------------------

        # remove a host
        # update the host to (1 0 0) from (1 1 0)
        # add a 5% failure probability
        # remove should fail if privileged access achieved
        # check access on the original state not the updated state
        is_remove = np.logical_and(action_alloc.reshape(-1) == 2, 1 - is_sleep)
        if np.any(is_remove):

            valid = np.logical_and(action_filter, is_remove)
            remove_success = np.random.choice([0, 1], size=valid.shape[0], p=[0.05, 0.95])

            valid = np.logical_and(valid, remove_success)
            host = host_alloc[valid]

            # check if the previous state is privileged

            # mod_state = orig_state[valid].copy().reshape(-1, len(HOSTS), 3)

            mod_state = next_state[valid].copy().reshape(-1, len(HOSTS), 3)
            is_priv = np.zeros((valid.shape[0]))
            is_priv[valid] = (mod_state[np.arange(len(host)), host, -1] == 1).reshape(-1)

            # check if red agent has compromised in this turn
            mod_next_state = next_state[valid].copy().reshape(-1, len(HOSTS), 3)
            is_next_priv = np.zeros((valid.shape[0]))
            is_next_priv[valid] = (mod_next_state[np.arange(len(host)), host, -1] == 1).reshape(-1)
            is_priv = np.logical_or(is_priv, is_next_priv)

            valid = np.logical_and(valid, 1 - is_priv)
            if np.any(valid):
                host = host_alloc[valid]
                # mod_state = orig_state[valid].copy().reshape(-1, len(HOSTS), 3)
                mod_state = next_state[valid].copy().reshape(-1, len(HOSTS), 3)
                is_scanned = (mod_state[np.arange(len(host)), host, 0] == 1)
                is_known = (mod_state[np.arange(len(host)), host, 0] >= 0)

                if np.any(is_known):
                    mod_state[np.arange(len(host)), host] = np.array([0, 0, 0])
                    if np.any(is_scanned):
                        scan_state = mod_state.copy()
                        scan_state[np.arange(len(host)), host] = np.array([1, 0, 0])
                        mod_state[is_scanned] = scan_state[is_scanned]
                    next_state[valid] = mod_state.reshape(mod_state.shape[0], -1)

                success[valid] = 1

        # host restoration ------------------------------------------

        # restore a host
        # update the host to (1 0 0) from (1 0 1)
        # add a 5% failure probability
        # check access on the original state not the updated state
        is_restore = np.logical_and(action_alloc.reshape(-1) == 3, 1 - is_sleep)
        if np.any(is_restore):
            valid = np.logical_and(action_filter, is_restore)
            restore_success = np.random.choice([0, 1], size=valid.shape[0], p=[0.0, 1.0])
            valid = np.logical_and(valid, restore_success)
            if np.any(valid):
                host = host_alloc[valid]
                # mod_state = orig_state.copy().reshape(-1, len(HOSTS), 3)[valid]
                mod_state = next_state[valid].copy().reshape(-1, len(HOSTS), 3)
                is_scanned = (mod_state[np.arange(len(host)), host, 0] == 1)
                is_known = (mod_state[np.arange(len(host)), host, 0] >= 0)
                is_exploited = (mod_state[np.arange(len(host)), host, 1] == 1)
                is_priv = (mod_state[np.arange(len(host)), host, 2] == 1)
                user_access = np.logical_or(is_priv.reshape(-1), is_exploited.reshape(-1))

                if np.any(user_access):
                    if np.any(is_known):
                        mod_state[np.arange(len(host)), host] = np.array([0, 0, 0])
                        if np.any(is_scanned):
                            scan_state = mod_state.copy()
                            scan_state[np.arange(len(host)), host] = np.array([1, 0, 0])
                            mod_state[is_scanned] = scan_state[is_scanned]
                        next_state[valid] = mod_state.reshape(mod_state.shape[0], -1)

                        # compute the negative cost of action
                # penalise due to disruption of network
                # remove impacting on host
                action_reward[valid] -= 1

                # update the decoys
                decoy_copy = decoy_reset.copy()[valid]
                decoy_copy[np.arange(len(host)), host] = 1
                decoy_reset[valid] = decoy_copy

                # reset impact
                impact_copy = new_impacted.copy()[valid]
                impact_copy[np.arange(len(host)), host] = 0
                new_impacted[valid] = impact_copy
                success[valid] = 1

    return next_state, action_reward, new_decoys, new_processes, success, decoy_reset, new_impacted, femitter_placed


class SimplifiedCAGE:
    '''
    A simplified version of the CAGE 2 Challenge environment 
    with faster execution speed and parallelism.
    '''

    def __init__(self, num_envs, num_nodes=13, remove_bugs=False):

        # basic parameters
        self.num_envs = num_envs
        self.num_nodes = num_nodes
        self.remove_bugs = remove_bugs

        # map integer in host_alloc[valid] exes to action name
        self.action_mapping = action_mapping()

        # reset all the parameters
        self.reset()



    def _set_init(
            self, num_envs, num_nodes, decoys=None, impacted=None,
            state=None, current_processes=None, detection=None):
        '''Set the initialisation parameters.'''

        # map host allocation to subnet
        # identify host priority
        self.subnets = np.tile(
            create_subnets(num_nodes).reshape(1, -1), (num_envs, 1))
        self.host_priority = np.tile(
            get_host_priority(HOSTS).reshape(1, -1), (num_envs, 1))

        # decoy and exploit information
        # -> given exploit index return compatible decoys
        # -> for each host return built in exploits
        # -> for each host return compatible decoys
        self.exploit_map = exploits_to_decoys(remove_bugs=self.remove_bugs)
        self.default_exploits = default_host_exploits(remove_bug=self.remove_bugs)
        self.default_decoys = np.tile(
            np.expand_dims(default_defender_decoys(),
                           axis=0), (self.num_envs, 1, 1))

        # set the initial state
        # add a privleged access to user0
        self.state = state
        if state is None:
            self.state = -np.ones((num_envs, num_nodes * 3))
            self.state[:, 24:27] = np.array([0, 0, 1])
        self.proc_states = None

        # keep track of action success
        self.blue_success = -np.ones((num_envs, 1))
        self.red_success = -np.ones((num_envs, 1))
        self.red_success = -np.ones((num_envs, 1))

        # keep track of impacts
        self.impacted = impacted
        if impacted is None:
            self.impacted = np.zeros((num_envs, num_nodes))

        # keep track of exploitable process and available decoys
        # legit process are marked as 1, decoys are -1
        # decoys are numbered  by priority for highest to lowest
        self.current_processes = current_processes
        if current_processes is None:
            self.current_processes = np.tile(
                np.expand_dims(self.default_exploits.copy(),
                               axis=0), (num_envs, 1, 1))

        # add placeholder selected exploit
        self.selected_exploit = -np.ones(num_envs)

        # log the decoys
        self.current_decoys = decoys
        if decoys is None:
            self.current_decoys = self.default_decoys.copy()

        # keep track of previously failed detection
        self.detection = detection
        if self.detection is None:
            self.detection = np.zeros((
                num_envs, num_nodes)).astype(bool)

        # get blue observation of the state
        state = self._process_state(
            state=self.state,
            logged_decoys=self.current_decoys)

        # keep track of the exploits used
        self.exploit_rewards = np.tile(
            construct_exploit_rew()[None], (num_envs, 1, 1))
        self.host_exploits = -np.ones((num_envs, num_nodes))

        # in bugged version femitter is stuck after being placed
        self.femitter_placed = np.zeros((
            num_envs, num_nodes)).astype(bool)

        return state

    def _get_info(self):
        info = {
            'impacted': self.impacted,
            'current_processes': self.current_processes,
            'current_decoys': self.current_decoys}
        return info

    def reset(self):

        # get the red and blue observation
        state = self._set_init(
            num_envs=self.num_envs,
            num_nodes=self.num_nodes)
        info = self._get_info()
        info["after_red_reward"] = np.zeros((self.num_envs, 1))
        info["Hosts_compromised_after_red"] = np.zeros((self.num_envs, 1))
        info["Hosts_compromised_after_blue"] = np.zeros((self.num_envs, 1))

        return state, info

    def step(self, red_action, blue_action, red_agent=None):
        err_msg = 'Ensure batch size is correct.'
        assert red_action.shape[0] == self.num_envs, err_msg
        assert blue_action.shape[0] == self.num_envs, err_msg

        # modify the state based on the actions
        # 1.0s over 10_000
        true_state, after_red_state, reward = self._process_actions(
            self.state, red_action, blue_action, self.subnets, red_agent)
        self.state = true_state.copy()

        # update the reward based on access
        # 0.001s over 10_000
        after_red_reward = self._process_reward(after_red_state, reward, self.impacted)
        # Hosts compromised after red action
        # hosts_compromised_red = self._process_intra_compromised(after_red_state)
        reward = self._process_reward(true_state, reward, self.impacted)
        done = np.zeros((true_state.shape[0], 1))

        # process the state for different observers
        # log the processed states 
        # 0.01s over 10_000
        next_state = self._process_state(
            state=true_state, logged_decoys=self.current_decoys,
            red_action=red_action, blue_action=blue_action)
        self.proc_states = next_state
        info = self._get_info()

        return next_state, reward, done, info

    def _process_intra_compromised(self, state):
        '''
        Process the intra-compromised hosts.
        '''

        # extract general information
        # mask out user0
        state_info = state.reshape(-1, self.num_nodes, 3).copy()
        state_info[:, 8] = 0
        # print(f"state_info: {state_info}")

        user_access = state_info[:, :, 1].reshape(-1) > 0
        # print(f"first user_access: {user_access} and sum of: {np.sum(user_access)}")
        priv_access = state_info[:, :, 2].reshape(-1) > 0
        # print(f"first priv_acess: {priv_access} and sum of: {np.sum(priv_access)}")

        priv_access = np.logical_or(user_access, priv_access)
        # print(f"2nd Privileged access: {priv_access}")
        # Number of hosts compromised
        hosts_comp = np.sum(priv_access)

        return hosts_comp

    def get_mask(self, state, decoys):
        '''
        Get the action mask for the current state configuration.
        '''

        # get the blue mask
        blue_mask = check_blue_action(state, decoys)

        # get the red action mask
        known, scanned, user, priv, subnet = check_red_access(state)
        red_mask, _ = get_possible_red_actions(
            user, priv, known, subnet, scanned)

        return {'Red': red_mask, 'Blue': blue_mask}

    def _process_actions(
            self, state, red_action, blue_action, subnets, red_agent=None):
        '''
        Update the internal states based on blue/red actions
        '''

        ############################################
        # TODO: are the success values usable  
        # -> red success is the only important one

        # -> success can be cancelled out via restore
        #   -> if restore occurs with priv
        ############################################

        # get next state and corresponding reward
        # add probability of failure
        red_true_state, red_reward, success, impacted, selected_exploit = update_red(
            state=state, action=red_action, subnet_loc=subnets,
            processes=self.current_processes,
            impacted=self.impacted,
            femitter_placed=self.femitter_placed,
            remove_bug=self.remove_bugs)
        self.red_success = success
        # print(f"Mid-step red success: {self.red_success}")
        self.selected_exploit = selected_exploit

        # update the host exploits
        host_selected = ((red_action - 4) % self.num_nodes).astype(int)
        self.host_exploits[
            np.arange(len(host_selected)), host_selected] = selected_exploit

        # now perform blue update
        # perform the blue action first
        true_state, blue_reward, decoys, proc, success, decoy_reset, impacted, femitter_placed = update_blue(
            state=state, updated_state=red_true_state,
            action=blue_action,
            decoys=self.current_decoys,
            processes=self.current_processes,
            proc_map=self.exploit_map,
            impacted=impacted,
            femitter_placed=self.femitter_placed
        )
        self.blue_success = success
        self.impacted = impacted
        self.femitter_placed = femitter_placed

        # reset the decoys
        if np.any(decoy_reset):
            decoys[decoy_reset.astype(bool)] = self.default_decoys[
                decoy_reset.astype(bool)]
            def_exploits = np.tile(
                self.default_exploits[None], (decoy_reset.shape[0], 1, 1))
            proc[decoy_reset.astype(bool)] = def_exploits[decoy_reset.astype(bool)]
        self.current_processes = proc
        self.current_decoys = decoys

        # impact action should also influence blue but negatively
        blue_reward -= red_reward

        return true_state, red_true_state, {'Blue': blue_reward, 'Red': red_reward}

    def _process_reward(self, state, action_reward, impacted):
        '''
        Calculate reward from updated state and combine
        with action-specific reward.
        '''

        # Location specific:
        # --------------------
        # user access 0.1
        # enterprise access 1
        # operational server 1
        # operational hosts 0.1

        # Action specific:
        # -----------------
        # opserver impact 10
        # blue restore 1

        #################################
        # NOTE: specific to configuraton
        #################################

        # extract general information
        # mask out user0
        state_info = state.reshape(-1, self.num_nodes, 3).copy()
        state_info[:, 8] = 0

        user_access = state_info[:, :, 1].reshape(-1) > 0
        priv_access = state_info[:, :, 2].reshape(-1) > 0

        flat_host = self.host_priority.reshape(-1)
        reward = np.zeros((state.shape[0], 1))

        # get the reward for each host
        host_exploits = self.host_exploits.reshape(-1)
        exploit_rewards = self.exploit_rewards.reshape(-1, len(EXPLOITS))
        valid_rewards = exploit_rewards[np.arange(len(host_exploits)),
        host_exploits.astype(int)]
        user_access = (user_access * valid_rewards).astype(bool)
        priv_access = np.logical_or(user_access, priv_access)

        # assign reward for privileged access to user/op hosts
        user_host_access = np.logical_and(priv_access, flat_host == 1)
        if len(user_host_access) > 0:
            user_host_rew = np.sum(
                user_host_access.reshape(-1, self.num_nodes), axis=-1)
            reward += user_host_rew.reshape(-1, 1) * 0.1

        # assign reward for privileged access ent/opserver
        ent_access = np.logical_and(
            priv_access, np.logical_or(flat_host == 2, flat_host == 3))
        if len(ent_access) > 0:
            ent_rew = np.sum(
                ent_access.reshape(-1, self.num_nodes), axis=-1)
            reward += ent_rew.reshape(-1, 1) * 1

        ###################################################
        # NOTE: specific to configuration
        # only care about impact to the operational server
        ###################################################

        # add impact reward
        # only interested in operational server
        reward += impacted[:, [7]] * 10

        # update the reward based on access
        action_reward['Blue'] = action_reward['Blue'] - reward
        action_reward['Red'] = action_reward['Red'] + reward

        return action_reward

    def _process_state(
            self, state, logged_decoys, red_action=None, blue_action=None):
        '''
        Convert the true state into observations of each agent.
        '''

        ##############################################
        # TODO: host should analyse unless otherwise
        ##############################################

        ##################################################
        # TODO: host should say removed unless otherwise
        ##################################################


        # add success to red state
        red_state = np.concatenate(
            [self.red_success.reshape(-1, 1), state], axis=-1)

        # initialise blank state
        blue_state = np.zeros((state.shape[0], 6 * self.num_nodes))
        activity_info = np.zeros((state.shape[0], self.num_nodes, 2))
        safety_info = np.zeros((state.shape[0], self.num_nodes, 2))
        scan_info = np.zeros((state.shape[0], self.num_nodes))
        decoy_info = np.zeros((state.shape[0], self.num_nodes))

        # extract the previous processed blue state
        # divide into scan activity, host safety, prior scans and decoy info
        if self.proc_states is not None:
            prev_state = self.proc_states['Blue']
            def_info = prev_state[:, :self.num_nodes * 4]
            added_info = prev_state[:, self.num_nodes * 4:]
            def_info = def_info.reshape(
                def_info.shape[0], self.num_nodes, -1)
            activity_info = def_info[:, :, :2]
            safety_info = def_info[:, :, 2:]
            scan_info = added_info[:, :self.num_nodes]
            decoy_info = added_info[:, self.num_nodes:]

        # add a green action
        # 50% chance to scan a host
        green_mask = np.zeros((activity_info.shape[0], self.num_nodes))
        selected_green_host = np.ones(activity_info.shape[0]) * -1
        green_scan = np.random.choice(
            [0, 1], size=activity_info.shape[0], p=[0.5, 0.5])
        if np.any(green_scan):
            green_host = np.random.choice(
                [1, 2, 3, 8, 9, 10, 11, 12], size=np.sum(green_scan))
            temp_activity = activity_info[green_scan.astype(bool)].copy()
            temp_activity[np.arange(len(green_host)), green_host.astype(int)] = np.array([1, 0])
            green_mask[np.arange(len(green_host)), green_host.astype(int)] = 1
            activity_info[green_scan.astype(bool)] = temp_activity
            selected_green_host[green_scan.astype(bool)] = green_host

        # determine consequences of red actions
        # also to a lesser extent the green actions
        if red_action is not None:

            # update prior activity
            # remove previous scans and other activity
            prev_scanned = np.logical_and(
                activity_info[:, :, 0].astype(bool), np.invert(green_mask.astype(bool)))
            if np.any(prev_scanned):
                temp_info = activity_info[np.any(prev_scanned, axis=-1)].copy().reshape(-1, 2)
                temp_info[:] = np.array([0, 0])
                temp_info = temp_info.reshape(-1, self.num_nodes, 2)
                activity_info[np.any(prev_scanned, axis=-1)] = temp_info

            # extract the host information
            host_alloc = ((red_action - 4) % self.num_nodes).reshape(-1).astype(int)

            # add in previously exploited hosts
            prev_exploited = state.reshape(-1, self.num_nodes, 3)[:, :, 1] == 1
            prev_exploited = np.logical_or(
                prev_exploited, state.reshape(-1, self.num_nodes, 3)[:, :, 2] == 1)
            prev_exploited = np.logical_and(
                prev_exploited.astype(bool), self.detection.astype(bool))
            if np.any(prev_exploited):

                # check if exploit is current host
                exp_idxs = np.where(prev_exploited)
                is_host = host_alloc[exp_idxs[0]] == exp_idxs[1]
                is_host = np.logical_or(is_host, exp_idxs[1] == 8)

                # ensure this is not the current exploit
                # or if it is user0 as this shouldn't appear
                curr_exploit = np.logical_and(
                    red_action > 3 + self.num_nodes,
                    red_action < (self.num_nodes * 2 + 4)).reshape(-1)
                curr_exploit = curr_exploit[exp_idxs[0]]
                ignore_exp = np.logical_and(is_host, np.invert(curr_exploit))
                ignore_exp = np.logical_or(ignore_exp, exp_idxs[1] == 8)
                if np.any(ignore_exp):
                    prev_exploited[exp_idxs[0][ignore_exp], exp_idxs[1][ignore_exp]] = False

            if np.any(prev_exploited):
                safety_copy = safety_info[np.any(prev_exploited, axis=-1)].copy()
                exp_idxs = np.where(prev_exploited[np.any(prev_exploited, axis=-1)])
                safety_copy[exp_idxs[0], exp_idxs[1]] = np.array([0, 1])
                safety_info[np.any(prev_exploited, axis=-1)] = safety_copy

            # only scan and exploits trigger alert
            # set scan to [1, 0, 0, 0]
            # add 5% chance it wont be recorded -> now 100%
            scan = np.logical_and(
                red_action > 3, red_action < (self.num_nodes + 4)).reshape(-1)
            scan_mask = np.random.choice([0, 1], size=scan.shape, p=[0.0, 1.0])
            scan = np.logical_and(scan, scan_mask)
            if np.any(scan):
                host = host_alloc[scan]
                activity_copy = activity_info[scan].copy()
                activity_copy[np.arange(len(host)), host] = np.array([1, 0])
                activity_info[scan] = activity_copy

            # check for exploits
            # set to [1, 1, 0, 1] if exploited
            # add 5% chance it goes unrecorded
            # if exploit is sshbruteforce this has 100% visibility
            # if green scanning with ssh -> access granted
            exploit = np.logical_and(
                red_action > 3 + self.num_nodes,
                red_action < (self.num_nodes * 2 + 4)).reshape(-1)

            exploit_mask = np.random.choice([0, 1], size=scan.shape[0], p=[0.05, 0.95])
            green_scanning = (selected_green_host == host_alloc).reshape(-1)
            ssh_selected = self.selected_exploit == (len(EXPLOITS) - 1)
            detected = np.logical_and(
                np.logical_or(exploit_mask, ssh_selected), np.invert(green_scanning))

            #####################################################
            # BUG: if host is ent0 -> always visible exploit
            # -> but undetectable if there is a decoy on host
            #####################################################
            decoys_on_ent0 = np.any(self.current_processes[:, 1] == -1, axis=-1)
            ent0_detected = np.logical_and((host_alloc == 1), np.invert(decoys_on_ent0))
            if not self.remove_bugs:
                detected = np.logical_or(detected, ent0_detected)
            exploit = np.logical_and(exploit, detected)

            # add action success
            # keep track of host detection
            exploit = np.logical_and(
                self.red_success.reshape(-1) == 1, exploit)

            self.detection[np.arange(len(host_alloc)), host_alloc] = exploit.astype(bool)
            if np.any(exploit):
                host = host_alloc[exploit]
                activity_copy = activity_info[exploit].copy()
                activity_copy[np.arange(len(host)), host] = np.array([1, 1])
                activity_info[exploit] = activity_copy
                safety_copy = safety_info[exploit].copy()
                safety_copy[np.arange(len(host)), host] = np.array([0, 1])
                safety_info[exploit] = safety_copy

        # determine consequences of blue actions
        if blue_action is not None:

            # extract the host action is applied to
            host_alloc = ((blue_action - 1) % self.num_nodes).reshape(-1).astype(int)

            # use analyse to reveal if privileged access achieved
            # [0, 0, 0, 0] to [0, 0, 1, 1]
            analyse = np.logical_and(
                blue_action > 0, blue_action < self.num_nodes + 1).reshape(-1).astype(bool)
            if np.any(analyse):
                host = host_alloc[analyse]
                mod_state = state.reshape(-1, self.num_nodes, 3)[analyse]

                is_priv = np.zeros_like(analyse)
                is_p = (mod_state[np.arange(len(host)), host, -1] == 1).reshape(-1)
                is_priv[analyse] = is_p
                analyse = np.logical_and(analyse, is_priv)

                host = host_alloc[analyse]
                safety_copy = safety_info[analyse].copy()
                safety_copy[np.arange(len(host)), host] = np.array([1, 1])
                safety_info[analyse] = safety_copy

            # remove processes from the network
            # -> if removed show [0, 0, 1, 0] -> however this is no guarantee it was removed
            remove = np.logical_and(
                blue_action > self.num_nodes * 2,
                blue_action < self.num_nodes * 3 + 1).astype(bool).reshape(-1)
            remove = np.logical_and(self.blue_success.reshape(-1), remove)
            if np.any(remove):
                host = host_alloc[remove]
                temp_safety = safety_info[remove].copy()
                temp_safety[np.arange(len(host)), host] = np.array([1, 0])
                safety_info[remove] = temp_safety

                # restore the machine to default
            # i.e. 0 0 0 0
            restore = (blue_action > self.num_nodes * 3).reshape(-1).astype(bool)
            restore = np.logical_and(restore, self.blue_success.reshape(-1))
            if np.any(restore):
                host = host_alloc[restore]
                temp_safety = safety_info[restore].copy()
                temp_safety[np.arange(len(host)), host] = np.array([0, 0])
                safety_info[restore] = temp_safety

                activity_copy = activity_info[restore].copy()
                activity_copy[np.arange(len(host)), host] = np.array([0, 0])
                activity_info[restore] = activity_copy

        # extract scan information
        # reset prior highlighted scans            
        # highlight the most recent scans 
        scan_info[(scan_info == 2)] = 1
        scan_info += activity_info[:, :, 0] * 2
        scan_info = np.clip(scan_info, a_min=0, a_max=2)
        # print(f'this is the scan info: {scan_info}')

        # set the decoy information
        decoy_info = logged_decoys.copy()
        decoy_info = np.sum(decoy_info > 0, axis=-1)
        decoy_info = decoy_info.reshape(scan_info.shape[0], -1)
        # print(f'this is the decoy info: {decoy_info}')

        # combine all the components 
        default_state = np.concatenate([
            activity_info, safety_info], axis=-1)

        # print(f'this is the default state: {default_state}')

        blue_state = np.concatenate([
            default_state.reshape(scan_info.shape[0], -1),
            scan_info, decoy_info], axis=-1)

        obs_state = {
            'Red': red_state, 'Blue': blue_state
        }

        # print(f'this is the blue state: {blue_state}')

        return obs_state

    def set_game_state(
            self, state, impacted, current_processes, current_decoys, detection):
        '''Set the state of the environment.'''

        # update the environmental number and configuration
        self.num_envs = state.shape[0]
        self.num_nodes = state.shape[-1] // 3

        # reset the necessary parameters
        state = self._set_init(
            num_envs=self.num_envs, num_nodes=self.num_nodes,
            decoys=current_decoys, state=state, impacted=impacted,
            current_processes=current_processes, detection=detection)

