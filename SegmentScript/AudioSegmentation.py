import os
import sys
from AudioSegmentationUtilities import CreateFolderStructure, GetAudioFiles, ParseXMLFile, SegmentAudioFiles #, moveFileToDigitPath

class AudioSegmentation(object):
    def __init__(self, folderPath):

        self.folderPath = folderPath

    def PerformAudioSegmentation(self):
        #get list of wav files using glob
        #create folder structure
        rootFolder = os.path.join(os.path.dirname(__file__), "S0386")
        if not os.path.exists(rootFolder):
            CreateFolderStructure(rootFolder)
        audioFiles = GetAudioFiles(self.folderPath)
        # print (audioFiles)

        #get corresponding xml files
        for audioFile in audioFiles:
            xmlFile = os.path.splitext(audioFile)[0] + '.xml'
            objSequence = ParseXMLFile(xmlFile)


            strDigitsInXML = os.path.splitext(audioFile)[0].rsplit('-', 1)[1]
            lstDigits = list(strDigitsInXML)
            digitById = objSequence.digitsById
            digitsBySegmentLen = {}
            for digitNo in lstDigits:
                for digit in digitById:
                    if str(digit.digitId) == digitNo:
                        #can use conditional if
                        if not digit.startTightDigit is None:
                            startDigit = digit.startTightDigit
                        else:
                            startDigit = digit.startDigit
                        if not digit.endTightDigit is None:
                            endDigit = digit.endTightDigit
                        else:
                            endDigit = digit.endDigit
                        # print ("startDigit: " + startDigit)
                        # print ("endDigit" + endDigit)
                        break
                    else:
                        continue
                digitsBySegmentLen[digitNo] = (startDigit, endDigit)
            # print (digitsBySegmentLen)

            segmentedFiles = SegmentAudioFiles(audioFile, digitsBySegmentLen)

if __name__ == "__main__":
    import sys
    audioSegmentation = AudioSegmentation((sys.argv[1]))
    audioSegmentation.PerformAudioSegmentation()
