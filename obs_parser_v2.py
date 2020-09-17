import numpy as np

class Obs():
    """Parse the observations from Halite."""

    def __init__(self, observation):
        """
        Init obs obj
        :param observation: observation from Halite
        """
        self.observation = observation
        self.me = observation['player']
        self.board = np.reshape(np.float32(observation['halite']), (1, 21, 21))
        self.halite_mean_v0 = np.mean(np.float32(observation['halite']))
        self.halite_mean_v1 = np.sum(np.float32(observation['halite'])) / np.sum(np.float32(observation['halite']) > 0)
        self.halite_75 = np.percentile(np.float32(observation['halite']), 75)
        self.halite_95 = np.percentile(np.float32(observation['halite']), 95)
        iterOrder = [0, 1, 2, 3]
        iterOrder[0], iterOrder[self.me] = iterOrder[self.me], iterOrder[0]
        self.collected = np.array([observation['players'][i][0]/5000. for i in iterOrder])
        self.incargo = np.zeros(4)
        self.shipNum = np.array([len(observation['players'][i][2])/20. for i in iterOrder])
        self.shipyardNum = np.array([len(observation['players'][i][1])/5. for i in iterOrder])
        self.step = observation['step']
        self.shipCargo = np.zeros((4, 21, 21))
        self.shipyard = np.zeros((4, 21, 21))
        self.shipPosDict = {}
        self.shipyardPosDict = {}
        for i, p in enumerate(iterOrder):
            for key, val in observation['players'][p][1].items():
                r, c = val // 21, val % 21
                self.shipyard[i, r, c] = 1
                if i == 0:
                    self.shipyardPosDict[(r, c)] = key
            for key, val in observation['players'][p][2].items():
                n, cargo = val
                r, c = n // 21, n % 21
                self.incargo[i] += cargo/5000.
                self.shipCargo[i, r, c] = (cargo+1000) / 1000.0
                if i == 0:
                    self.shipPosDict[(r, c)] = key

        self.matState = np.concatenate((self.board/1000, self.shipCargo, self.shipyard), axis=0)  ### 9 * 21 * 21
        self.vecState = np.array([self.step/400., self.halite_mean_v0/500., self.halite_mean_v1/500.,
                                  self.halite_75/500., self.halite_95/500.])
        self.vecState = np.concatenate((self.collected, self.incargo, self.shipNum, self.shipyardNum, self.vecState)) ### 1, 21
        self.state = (self.matState, self.vecState)
        self.Realized = self.collected[0] * 5000
        self.unRealized = self.incargo[0] * 5000
        
def generate_offset_map(Hmap, row_c, col_c):
    """
    Generate a view of map that centered at (N, row_c, col_c)
    :param map: old map centered at (10, 10)
    :param row_c: new center in row
    :param col_c: new center in col
    :return: new view of map
    """
    Hmap_copy = Hmap.copy()
    v_shift = row_c - 10
    h_shift = col_c - 10
    h_index = (np.arange(21) + h_shift) % 21
    v_index = (np.arange(21) + v_shift) % 21
    temp = Hmap_copy[:, v_index]
    
    return temp[:, :, h_index]

def actionToTensor(action_dict, shipPosDict, shipyardPosDict):
    """
    Return action tensor 0->N, 1->S, 2->W, 3->E, 4->Stay, 5->Conv, 6->Spawn, 7->Stay
    weight for each class: 0.01, 0.1, 0.1, 0.1, 0.1, 1, 10, 10, 1
    :param action_dict:
    :param shipPosDict:
    :param shipyardPosDict:
    :return:
    """
    entity_action_dim = {"NORTH": 0,
                   "SOUTH": 1,
                   "WEST": 2,
                   "EAST": 3,
                   "STAY": 4,
                   "CONVERT": 5,
                   "NONE": 6,
                   "SPAWN": 7}
    
    entity_action_tensor = np.zeros((8, 21, 21))

    if len(shipPosDict) > 0:
        for pos, id in shipPosDict.items():
            if id not in action_dict:
                entity_action_tensor[4, pos[0], pos[1]] = 1
            else:
                entity_action_tensor[entity_action_dim[action_dict[id]], pos[0], pos[1]] = 1
    if len(shipyardPosDict) > 0:
        for pos, id in shipyardPosDict.items():
            if id not in action_dict:
                entity_action_tensor[6, pos[0], pos[1]] = 1
            else:
                entity_action_tensor[entity_action_dim[action_dict[id]], pos[0], pos[1]] = 1

    #ship_action_tensor = np.pad(ship_action_tensor, ((0, 0), (10, 10), (10, 10)), 'wrap')
    #shipyard_action_tensor = np.pad(shipyard_action_tensor, ((0, 0), (10, 10), (10, 10)), 'wrap')

    return entity_action_tensor

def logitToAction(entity_action_prob, isShip=True):
    
    entity_action_dim = {0: "NORTH",
                       1: "SOUTH",
                       2: "WEST",
                       3: "EAST",
                       4: "STAY",
                       5: "CONVERT",
                       6: "NONE",
                       7: "SPAWN"}
    
    #shipAction, shipyardAction = shipAction[:, 10:31, 10:31], shipyardAction[:, 10:31, 10:31]
    if isShip:
        ordered_actions = [(entity_action_dim[i], entity_action_prob[i]) for i in range(6)]
    else:
        ordered_actions = [(entity_action_dim[i], entity_action_prob[i]) for i in range(6,8)]
        
    ordered_actions = sorted(ordered_actions, key=lambda x: x[1], reversed=True)
    
    return ordered_actions
