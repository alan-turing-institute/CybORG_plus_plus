


<p align="center">
    <img src="Extras/images/logo_cyborg.png" alt="Diagram of the system" width="400"/>
</p>

**CybORG++ is a toolkit for reinforcement learning research focused on autonomous network defense.**


Building on the CAGE 2 CybORG environment, it introduces key improvements, including enhanced debugging capabilities, refined agent implementation support, and a streamlined environment that enables faster training and easier customization. Along with addressing several software bugs from its predecessor, CybORG++ introduces MiniCAGE, a lightweight version of CAGE 2, which improves performance dramatically—up to 1000× faster execution in parallel iterations—without sacrificing accuracy or core functionality. 

This repository contains:

- A debugged version of the CAGE 2 CybORG environment that is compatible with the CAGE 2 challenge
  - [Debugged Version of CAGE 2 CybORG](./Debugged_CybORG)
- A simple and fast reimplementation of the CAGE 2 CybORG environment
  - [Mini CAGE 2 CybORG Reimplementation](./mini_CAGE)
- Additional information on how the environment works and things we’ve learnt that may be helpful for future users

If you use this repository in your research, please cite it as follows:

```bibtex
@article{emerson2024cyborg++,
  title={Cyborg++: An enhanced gym for the development of autonomous cyber agents},
  author={Emerson, Harry and Bates, Liz and Hicks, Chris and Mavroudis, Vasilios},
  journal={arXiv preprint arXiv:2410.16324},
  year={2024}
}
```

# Extended CAGE 2 Developer Guide

This guide extends the previous CAGE 2 developer guide, highlighting features of the environment that are useful for successful model implementation. 

The following info is to the best of our knowledge, and if there are intricacies that we have missed or miswritten then please feel free to contact either of this Repo’s owners and we will discuss and update.

## Network Diagram


### **True Network Diagram**

<p align="center">
    <img src="Extras/images/True_Network_Diagram.png" alt="Diagram of the system" width="600"/>
    <figcaption style="text-align: center;"><em>Figure 1: CAGE 2 CybORG Network Diagram. The orange dotted line 
indicates a shared firewall between the User subnet and Enterprise subnet. The red dotted line indicates the defender is not a stationary host in the network, and that User0 is where red maintains a foothold on the system whilst not functioning as a proper user host.
</em></figcaption>
</p>

## CybORG Agents

The red agents will start from one of the user host machines 1-4. There is a fifth user host (user host 0) but this cannot be acted upon from the blue agent, it seems this is how the red agent will always have a foothold on the user subnet. If the red agent attacks (DNS, exploits and privilege escalates) a user machines 1 or 2 successfully then it will move to Enterprise host 1.  If the red agent attacks (DNS, exploits and privilege escalates) user machines 3 or 4 successfully then it will move to Enterprise host 0. From here, the red agent is able to move to Enterprise host 2 (from either Enterprise host 0 or 1), given another successful attack. Once at Enterprise host 2, it can then directly access the Operational Server if the attack is successful and then begin Impacting this host, causing the -10 penalty for the blue agent. 
This red behaviour pattern can be seen in the [scenario2.yaml](./Debugged_CybORG/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml) file and [b_line.py](./Debugged_CybORG/CybORG/CybORG/Agents/SimpleAgents/B_line.py) files. You are able to see the information each host has access to, and also what kind of machines they are (linux or windows).

The B-line agent moves through the network with domain knowledge of the network already, so it is aware of how to migrate to the Operational Server and impact in the quickest number of steps. 
It’s of note here that the Operational hosts 0, 1 and 2 are never interacted with at all in the b-line agent trajectory since the operational server is always reached first, and this is the goal.
The meander agent, on the other hand, prioritises exploring each subnet before moving into the next. The meander agent follows an action hierarchy of impact, scan subnet, scan host, escalate and exploit, and will always select actions with higher priority first. Similarly, within these categories actions that have been made available earliest will also be prioritised (i.e. if user1 is scanned first then this will be the first host to be exploited).   


## Action Space

The blue agent can take one of several actions in each timestep: 
- Analyse a host - reveals with 100% certainty the presence of an exploited host
- Remove host - remove low level access users from a host
- Restore host - remove privliged users from a host
- Place a decoy - set-up a decoy service on a host. If selected the attacker's action will fail.

One of the unique features of the CAGE 2 environment is that both agents take actions simultaneously, rather than sequentially. This means that to some degree an agent must attempt to anticipate the actions of its opponent, especially as some actions will have a nullifying effect if they occur at the same time. Below is a list of action priorities should they occur simultaneously:

| Red Agent Action | Blue Agent Action | Priority |
|------------------|-------------------|----------|
| Escalate           | Remove            | Escalate |
| Exploit            | Decoy             | Exploit  |
| Exploit            | Restore           | Exploit  |
| Escalate | Restore           | Restore  |


**Method of taking actions**

The way actions are taken is not immediately clear. However, on examination it seems as though both the red and blue agents take an action in a single turn both based on the outcome of the state prior to the step.

First the red agent takes an action based on the observation of s, then the blue agent take an action also based on s with no knowledge of the red action that’s just been decided. This means that the blue agent wastes a step potentially as it doesn’t see the outcome of the red action until the next step, where the red could do something like DRS and get a foothold in the next subnet. 

It seems like when actions happen which would result in a conflicting observation space after the step, the blue agent’s action is prioritised. An example is if the red agent tries to privilege escalate a host but the blue agent decides to restore that same host in the same time step. These are mutually exclusive actions but they could both be possible to do according to s, but we can’t have a host that is both fully compromised and restored, so Blue’s restore action will take priority in a case like this. 

**The Remove Action**

The remove action was always returned as a successful action, despite whether it actually was or not (see issue 1 in [Debugged Version of CAGE 2 CybORG](./Debugged_CybORG) - this was fixed). 

Another intricacy with the remove action seems to be with it only working for specific exploits on specific hosts. 
This is because the remove action only work if the usernames associated with the ports the exploits were occurring 
on were ‘root’ or ‘SYSTEM’ then the remove action would fail. This was the case for some exploits, so this means 
that remove will not always work for every exploit on every host. This may not be a bug, more of feature of the 
system. But we feel it is still worth mentioning to help understand any behaviour that may not make sense otherwise. 
The info in the Hosts to Default Processes,Ports and users Mapping Table in the Appendix may be of use if a remove action is not working as expected, as it may be due to the user being 'root' or 'SYSTEM' for that process on that port.


## Observation Space

The blue agent observation space is processed using the Wrappers available in [Wrappers](./Debugged_CybORG/CybORG/CybORG/Agents/Wrappers/) and the default wrapper is displayed in ChallengeWrapper.py. The blue agent observation has 52 features, with each four consecutive features corresponding to a single node in the network for a total of 13 nodes (in order: Def, Ent0, Ent1, Ent2, OpHost0, OpHost1, OpHost2, Opserver, User0, User1, User2, User3, User4). Each four feature can be one of five combinations describing the node's state:
- [0, 0, 0, 0] = No activity detected on host
- [1, 0, 0, 0] = Scan detected on host
- [1, 0, 1, 1] = Exploit detected in previous turn
- [0, 0, 1, 1] = Exploit detected in prior turns
- [0, 0, 1, 0] = Remove action applied to host (does not indicate successful removal)

Scans and removal actions have 100% chance of being observed in the state, however for exploitation the probability is less clear. Exploitation actions have a 95% chance of being observed with a few exceptions (details for this can be found in [ExploitAction.py](./Debugged_CybORG/CybORG/CybORG/Shared/Actions/ConcreteActions/ExploitAction.py). SSHBruteForce always has a 100% chance of being observed, consequently an exploit on the OpServer, which can only be exploited using this method, will always be represented in the state. 

## Reward Signal

Reward is assigned based on three conditions: exploiting hosts, restoring hosts and impacting critical hosts. Each host is assigned a numeric value corresponding to its importance (ConfidentialityValue in [Scenario2.yaml](/home/harry/Documents/cyber/BlueTeam/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml)). When a host is exploited this specifies the amount of reward provided: -0.1 for UserHosts and OpHosts, -1 for EntHosts and the OpServer. Using the restore action gives a reward of -1 regardless of which host it is applied to. The only host in which the impact action yields any reward is the OpServer. Allowing this host to be impacted results in a reward of -10. This reward persists until the operational server has been restored and need not be applied in every timestep as the logic of the pre-programmed agents may suggest. The total reward for each timestep is then the sum of exploited host rewards, hosts restored in that timestep and hosts currently being impacted. 

### Appendix

### Decoy Deployment Strategy

| Host     | OS      | Local Ports                  | Decoys                     | Decoy Order                                                    | Initial Exploit Order                               |
|----------|---------|-----------------------------|----------------------------|----------------------------------------------------------------|-----------------------------------------------------|
| **User0** | Windows | 21,<br>22                    | Apache, Smss,<br>Svchost, Tomcat  | Svchost - 1 <br> Smss - 2 <br> Apache - 3 <br> Tomcat - 4      | FTPDirTraversal (port 21) : 7 <br> SSHBruteForce (port 22) : 0.1 |
| **User1** | Windows | 21,<br>22                    | Apache, Smss,<br>Svchost, Tomcat  | Svchost - 1 <br> Smss - 2 <br> Apache - 3 <br> Tomcat - 4      | FTPDirTraversal (port 21) : 7 <br> SSHBruteForce (port 22) : 0.1 |
| **User2** | Windows | 445,<br>139,<br>135,<br>3389 | Apache, SSHd,<br>Tomcat, Femitter | SSHd - 0.1 <br> Apache - 3 <br> Tomcat - 4 <br> Femitter - 7  | EternalBlue (port 139) : 2 <br> BlueKeep (port 3389) : 1 |
| **User3** | Linux   | 25,<br>80,<br>443,<br>3390   | SSHd, Vsftpd                | SSHd - 0.1 <br> Vsftpd - 7                                     | HarakaRCE (port 25) : 6 <br> SQLInjection (port 3390 and 80/443) : 5 <br> HTTPsRFI (port 443) : 4 <br> HTTPRFI (port 80) : 3 <br> BlueKeep (port 3389) : 1 |
| **User4** | Linux   | 22,<br>80,<br>3390,<br>443,<br>25 | Vsftpd                    | Vsftpd - 7                                                     | HarakaRCE (port 25) : 6 <br> SQLInjection (port 3390 and 80/443) : 5 <br> HTTPsRFI (port 443) : 4 <br> HTTPRFI (port 80) : 3 <br> BlueKeep (port 3389) : 1 |
| **Ent0** | Linux   | 22                          | Apache, Tomcat,<br>Vsftpd, HarakaSMPT | Apache - 3 <br> Tomcat - 4 <br> Vsftpd - 7 <br> HarakaSMPT - 6 | SSHBruteForce (port 22) : 0.1 |
| **Ent1** | Windows | 22,<br>135,<br>3389,<br>445,<br>139,<br>80,<br>443 | Femitter         | Femitter - 7                                                  | HTTPsRFI (port 443) : 4 <br> HTTPRFI (port 80) : 3 <br> EternalBlue (port 139) : 2 <br> BlueKeep (port 3389) : 1 <br> SSHBruteForce (port 22) : 0.1 |
| **Ent2** | Windows | 22,<br>135,<br>3389,<br>445,<br>139,<br>80,<br>443 | Femitter         | Femitter - 7                                                  | SSHBruteForce (port 22) : 0.1 |
| **Op_host0** | Linux| 22                          | Apache, HarakaSMPT,<br>Tomcat, Vsftpd | Vsftpd - 7 <br> HarakaSMPT - 6 <br> Tomcat - 4 <br> Apache - 3 | SSHBruteForce (port 22) : 0.1 |
| **Op_host1** | Linux| 22                          | Apache, HarakaSMPT,<br>Tomcat, Vsftpd | Vsftpd - 7 <br> HarakaSMPT - 6 <br> Tomcat - 4 <br> Apache - 3 | SSHBruteForce (port 22) : 0.1 |
| **Op_host2** | Linux| 22                          | Apache, HarakaSMPT,<br>Tomcat, Vsftpd | Vsftpd - 7 <br> HarakaSMPT - 6 <br> Tomcat - 4 <br> Apache - 3 | SSHBruteForce (port 22) : 0.1 |
| **Op_Server** | Linux| 22                        | Apache, HarakaSMPT,<br>Tomcat, Vsftpd | Vsftpd - 7 <br> HarakaSMPT - 6 <br> Tomcat - 4 <br> Apache - 3 | SSHBruteForce (port 22) : 0.1 |

In the decoy order column, the numbers that are after each decoy refer to the ranking of the decoy in the order of priority. The lower the number, the higher the priority and this is hardcoded in the exploit_options dictionary in 
the `Debugged_CybORG/CybORG/CybORG/Shared/Actions/AbstractActions/ExploitRemoteService.py` file.

### Exploits to decoy to process mapping table

| **Exploits**              | **Decoys**                       | **Processes** |
|---------------------------|----------------------------------|--------------|
| EternalBlue               | decoySmss                        | smss.exe     |
| BlueKeep                  | decoySvchost                     | svchost.exe  |
| HTTPRFI                   | decoyApache                      | apache2      |
| HTTPSRFI                  | decoyTomcat                      | tomcat8.exe  |
| SSH BruteForce            | decoySSHD                        | sshd.exe/sshd |
| SQL Injection             | -                                | mysql        |
| Haraka RCE                | decoyHarakaSMTP                  | smtp         |
| FTP Directory Traversal   | decoyFemitter, <br/> decoyVsftpd | femitter.exe |

### Hosts to Default Processes, Ports and users Mapping Table

| **Hostname** | **Ports**          | **Users**          | **Processes**        |
|-----------------------|--------------------|--------------------|----------------------|
| User1           | 22, 21             | SSHD_SERVER, SYSTEM | SSHD.EXE, FEMITTER.EXE |
| User2           | 445, 139, 135, 3389 | SYSTEM, SYSTEM, NETWORK | SMSS.EXE, SVCHOST.EXE, SVCHOST.EXE |
| User3          | 3389, 80, 443, 25  | ROOT, WWW-DATA, ROOT | MYSQL, APACHE2, SMTP |
| User4           | 22, 3390, 80, 443, 25 | ROOT, ROOT, WWW-DATA, ROOT | SSHD, MYSQL, APACHE2, SMTP |
| Ent0       | 22                 | ROOT               | SSHD.EXE              |
| Ent1       | 22, 135, 3389, 445, 139, 80, 443 | SSHD_SERVER, SYSTEM, SYSTEM, SYSTEM, NETWORK | SSHD.EXE, SVCHOST.EXE, SVCHOST.EXE, SMSS.EXE, TOMCAT8.EXE |
| Ent2       | 22, 135, 3389, 445, 139, 80, 443 | SSHD_SERVER, SYSTEM, SYSTEM, SYSTEM, NETWORK | SSHD.EXE, SVCHOST.EXE, SVCHOST.EXE, SMSS.EXE, TOMCAT8.EXE |
| Op_Server      | 22                 | ROOT               | SSHD                 |
| Op_host0      | 22                 | ROOT               | SSHD                 |
| Op_host1     | 22                 | ROOT               | SSHD                 |
| Op_host2     | 22                 | ROOT               | SSHD                 |
| Defender          | 22, 53, 78         | ROOT, SYSTEMD+     | SSHD, SYSTEMD        |

---
### License

CybORG++ is released under MIT license. Please see LICENSE for details.

