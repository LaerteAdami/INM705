class CityscapesClass():
    '''
    A container to store class ids and colors
    '''
    
    def __init__(self, case):
        
        if case == "4 classes":
            
            self.trainId = {
                    0: 0,  # unlabeled
                    1: 0,  # ego vehicle
                    2: 0,  # rect border
                    3: 0,  # out of roi
                    4: 0,  # static
                    5: 0,  # dynamic
                    6: 0,  # ground
                    7: 1,  # road
                    8: 0,  # sidewalk
                    9: 0,  # parking
                    10: 0,  # rail track
                    11: 0,  # building
                    12: 0,  # wall
                    13: 0,  # fence
                    14: 0,  # guard rail
                    15: 0,  # bridge
                    16: 0,  # tunnel
                    17: 0,  # pole
                    18: 0,  # polegroup
                    19: 0,  # traffic light
                    20: 0,  # traffic sign
                    21: 0,  # vegetation
                    22: 0,  # terrain
                    23: 2,  # sky
                    24: 0,  # person
                    25: 0,  # rider
                    26: 3,  # car
                    27: 0,  # truck
                    28: 0,  # bus
                    29: 0,  # caravan
                    30: 0,  # trailer
                    31: 0,  # train
                    32: 0,  # motorcycle
                    33: 0,  # bicycle
                    -1: 0  # licenseplate
                }
            self.colors = [(  0,  0,  0), # background
                               (128, 64,128), # road
                               ( 70,130,180), # sky
                               (  0,  0,142)  # car
                          ]
            
            self.labels = ["background", "road", "sky", "car"]
            
        elif case == "6 classes":
            self.trainId = {
                    0: 0,  # unlabeled
                    1: 0,  # ego vehicle
                    2: 0,  # rect border
                    3: 0,  # out of roi
                    4: 0,  # static
                    5: 0,  # dynamic
                    6: 0,  # ground
                    7: 1,  # road
                    8: 0,  # sidewalk
                    9: 0,  # parking
                    10: 0,  # rail track
                    11: 2,  # building
                    12: 0,  # wall
                    13: 0,  # fence
                    14: 0,  # guard rail
                    15: 0,  # bridge
                    16: 0,  # tunnel
                    17: 0,  # pole
                    18: 0,  # polegroup
                    19: 0,  # traffic light
                    20: 0,  # traffic sign
                    21: 3,  # vegetation
                    22: 0,  # terrain
                    23: 4,  # sky
                    24: 0,  # person
                    25: 0,  # rider
                    26: 5,  # car
                    27: 0,  # truck
                    28: 0,  # bus
                    29: 0,  # caravan
                    30: 0,  # trailer
                    31: 0,  # train
                    32: 0,  # motorcycle
                    33: 0,  # bicycle
                    -1: 0  # licenseplate
                }
            
            self.colors = [(  0,  0,  0), # background
                               (128, 64,128), # road
                               ( 70, 70, 70), # building
                               (107,142, 35), # vegetation                                    
                               ( 70,130,180), # sky
                               (  0,  0,142)  # car
                          ]
            
            self.labels = ["background", "road", "building", "vegetation", "sky", "car"]            
            
        elif case == "10 classes":
            self.trainId = {
                    0: 0,  # unlabeled
                    1: 0,  # ego vehicle
                    2: 0,  # rect border
                    3: 0,  # out of roi
                    4: 0,  # static
                    5: 0,  # dynamic
                    6: 0,  # ground
                    7: 1,  # road
                    8: 0,  # sidewalk
                    9: 0,  # parking
                    10: 0,  # rail track
                    11: 2,  # building
                    12: 0,  # wall
                    13: 0,  # fence
                    14: 0,  # guard rail
                    15: 0,  # bridge
                    16: 0,  # tunnel
                    17: 0,  # pole
                    18: 0,  # polegroup
                    19: 0,  # traffic light
                    20: 3,  # traffic sign
                    21: 4,  # vegetation
                    22: 0,  # terrain
                    23: 5,  # sky
                    24: 6,  # person
                    25: 0,  # rider
                    26: 7,  # car
                    27: 0,  # truck
                    28: 8,  # bus
                    29: 0,  # caravan
                    30: 0,  # trailer
                    31: 0,  # train
                    32: 0,  # motorcycle
                    33: 9,  # bicycle
                    -1: 0  # licenseplate
                }
            
            self.colors = [(  0,  0,  0), # background
                               (128, 64,128), # road
                               ( 70, 70, 70), # building
                               (220,220,  0), # traffic sign
                               (107,142, 35), # vegetation         
                               ( 70,130,180), # sky
                               (220, 20, 60), # person
                               (  0,  0,142), # car
                               (  0, 60,100), # bus
                               (119, 11, 32), # bicycle
                          ]
            
            self.labels = ["background", "road", "building", "traffic sign", "vegetation", 
                           "sky", "person", "car", "bus", "bicycle"]        
    