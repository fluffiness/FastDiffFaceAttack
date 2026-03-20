AGE_GENDER_RACE_MAP = {
    "age": {
        0: "teens", 
        1: "teens", 
        2: "twenties", 
        3: "thirties", 
        4: "forties", 
        5: "fifties", 
        6: "sixties", 
        7: "seventies"
    },
    "gender": {
        0: "man",
        1: "woman"
    },
    "race": {
        0: "white",
        1: "black",
        2: "asian",
        3: "indian",
        4: "ambiguous"
    }
}

ALT_AGE_GENDER_RACE_MAP = {
    "age": {
        0: "young", 
        1: "young", 
        2: "young", 
        3: "adult", 
        4: "middle-aged", 
        5: "middle-aged", 
        6: "old", 
        7: "old"
    },
    "gender": {
        0: "man",
        1: "woman"
    },
    "race": {
        0: "white",
        1: "black",
        2: "asian",
        3: "indian",
        4: "ambiguous"
    }
}

THRES_DICT = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
            'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878), 
            'cosface': (0.144840, 0.241045, 0.312703), 'arcface': (0.144840, 0.241045, 0.312703)}

KEY_WORDS = ["face", "man", "woman", "his", "her", "race", "racially", 
                 "white", "black", "asian", "indian","ambiguous", 
                 "teens", "twenties", "thirties", "forties", "fifties", "sixties", "seventies"]