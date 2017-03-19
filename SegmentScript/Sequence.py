class Sequence(object):
    def __init__(self):
        self.IdSpeaker = ''
        self.IdSession = ''
        self.device = ''
        self.typeSequence = ''
        #self.digits = '';
        self.digitsById = []

    def __init__(self, speakerId, sessionId,device, typeSequence,digitsByID):
        self.IdSpeaker = speakerId
        self.IdSession = sessionId
        self.device = device
        self.typeSequence = typeSequence
        #self.digits = digits;
        self.digitsById = digitsByID


if __name__ == "__main__":
    pass
