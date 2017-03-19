import os
import glob
import itertools
from xml.dom import minidom
import wave
import shutil

from Sequence import Sequence
from Digit import Digit
import xml.etree.ElementTree as ET
digitsByName = {'CERO':0, 'UNO':1,'DOS':2, 'TRES':3, 'CUATRO':4,'CINCO':5,'SEIS':6,'SIETE':7,'OCHO':8,'NUEVE':9}
def CreateFolderStructure(rootFolder):
    os.makedirs(rootFolder)
    for key, value in digitsByName.items():
        os.makedirs(os.path.join(rootFolder, key))

def GetAudioFiles(folderPath):
    configFiles = itertools.chain.from_iterable(glob.iglob(os.path.join(root,'*.wav')) for root, dirs, files in os.walk(folderPath))
    return list(configFiles)

def ParseXMLFile(xmlFileName):
    xmlDoc = ET.parse(xmlFileName)
    rootNode = xmlDoc.getroot()
    idSpeaker = rootNode.find('IdSpeaker').text
    idSession = rootNode.find('sessionId').text
    device = rootNode.find('Device').text
    typeSequence = rootNode.find('TypeSequence').text
    digits = rootNode.find('Digits').text
    digitsNode = rootNode.find('Digits')
    digitsById = []

    for digitNode in rootNode.findall('./Digits/digit'):
        print(digitNode)
        print ("digitNode")
        #check for null value
        id = digitNode.find('digit_number').text
        digitName = digitNode.find('digit_name').text
        startDigit = digitNode.find('start_digit').text
        endDigit = digitNode.find('end_digit').text
        startTightDigit = digitNode.find('start_tight_digit').text
        endTightDigit = digitNode.find('end_tight_digit').text
        digit = Digit(id, startDigit, endDigit, startTightDigit, endTightDigit,digitName)
        digitsById.append(digit)
    sequence = Sequence(idSpeaker, idSession,device, typeSequence,digitsById)
    return sequence

def SegmentAudioFiles(audioFile, digitsBySegmentLen):
    segmentedFiles = []
    origAudio = wave.open(audioFile, 'r')
    nframes = origAudio.getnframes()
    frameRate = origAudio.getframerate()
    nChannels = origAudio.getnchannels()
    sampWidth = origAudio.getsampwidth()
    dictFolderNames = {0:'CERO', 1:'UNO',2:'DOS', 3:'TRES', 4:'CUATRO',5:'CINCO',6:'SEIS',7:'SIETE', 8:'OCHO',9:'NUEVE'}

    print ("number of frames in the audio file =",nframes)
    print ("number of channels in the audio file =",nChannels)
    print ("frame rate = ", frameRate)

    for digitNumber, values in digitsBySegmentLen.items():
        startValue, endValue = int(values[0]), int(values[1])
        print ("startValue, endValue ", startValue, endValue )
        origAudio.setpos(startValue)
        chunkData = origAudio.readframes((endValue-startValue))
        origAudio.tell()
        segmentedFileName = os.path.splitext(audioFile)[0] + '-' + digitNumber + '.wav'
        rootPath = os.path.join(os.path.dirname(__file__), "S0386")
        filePath = os.path.join(rootPath, dictFolderNames[int(digitNumber)], os.path.basename(segmentedFileName))
        chunkAudio = wave.open(filePath,'w')
        chunkAudio.setnchannels(nChannels)
        chunkAudio.setsampwidth(sampWidth)
        chunkAudio.setframerate(frameRate)
        chunkAudio.writeframes(chunkData)
    chunkAudio.close()
    origAudio.close()
    return segmentedFiles
