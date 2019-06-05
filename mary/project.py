import numpy as np
import cv2
from midiutil import MIDIFile

def mapnote(note):
    if note == 0:
        return 40
    if note == 1:
        return 41
    if note == 2:
        return 43
    if note == 3:
        return 45
    if note == 4:
        return 47
    if note == 5:
        return 48
    if note == 6:
        return 50
    if note == 7:
        return 52
    if note == 8:
        return 53
    if note == 9:
        return 55
    if note == 10:
        return 57
    if note == 11:
        return 59
    if note == 12:
        return 60   
    if note == 13:
        return 62
    if note == 14:
        return 64
    if note == 15:
        return 65
    if note == 16:
        return 67
    if note == 17:
        return 69
    if note == 18:
        return 71
    if note == 19:
        return 72
    if note == 20:
        return 74
    if note == 21:
        return 76
    if note == 22:
        return 77
    if note == 23:
        return 79
    if note == 24:
        return 81
    if note == 25:
        return 83
def turnToMidi(musicls, connected):
    midi = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    pitch = 0
    duration = 1
    volume = 100
    
    midi.addTrackName(track,time,"test")
    midi.addTempo(track, time, 60)
    music = []
    for i, pitch in enumerate(musicls):
        if i in connected2 and pitch != musicls[i-1]:
            duration = 1/4
            midi.addNote(track,channel,pitch, time, duration, volume)
            time += 1/4
            music.append([pitch,1/4])
        elif i in connected and i+1 in connected2 and pitch != musicls[i+1]:
            duration = 3/4
            midi.addNote(track,channel,pitch, time, duration, volume)
            time += 3/4
            music.append([pitch,3/4])
        elif i in connected or i-1 in connected:
            duration = 1/2
            midi.addNote(track,channel,pitch, time, duration, volume)
            time += 1/2
            music.append([pitch,1/2])
        else:
            duration = 1
            midi.addNote(track,channel,pitch, time, duration, volume)
            time += 1
            music.append([pitch,1])
    print(music)
    midi_file = open("mary.mid", 'wb')
    midi.writeFile(midi_file)
    midi_file.close()
    
def getlinesgroup(sortlines):
    lastmarked =0
    kline = 0
    lines = []
    linegroup = []
    for i in range(len(sortlines)):
        if(sortlines[i][1] == sortlines[i][3] and sortlines[i][1] > header and sortlines[i][1] - lastmarked > smallgap):
            if kline == 4:
                lastmarked += gap
                kline = 0
                lines.append(sortlines[i][1])
                linegroup.append(lines)
                lines = []
                cv2.line(img, (sortlines[i][0], sortlines[i][1]), (sortlines[i][2], sortlines[i][3]), (0, 0, 255), 1, cv2.LINE_AA)
            else:
                lines.append(sortlines[i][1])
                lastmarked = sortlines[i][1]
                kline +=1
                cv2.line(img, (sortlines[i][0], sortlines[i][1]), (sortlines[i][2], sortlines[i][3]), (0, 0, 255), 1, cv2.LINE_AA)
    ##cv2.imwrite('houghlines5.jpg',img)
    return linegroup

def line7groupsFun(linegroup):
    line7groups = []
    for line in linegroup:
        l1 = line[0] - blockwide
        l7 = line[4] + blockwide
        line7groups.append([l7,line[4],line[3],line[2],line[1],line[0],l1])
    return line7groups


def getNotesAndTmpo(grayinv,imgg,line7groups,blockwide):
    notels=[]
    connected =[]
    connected2 =[]
    prevconnect = 0
    for l in line7groups:
        linesum = 0
        sumlist = []
        #notesx = []
        for i in range(len(grayinv[0])):
            linesum = 0
            for j in range(blocklen):
                #linesum += grayinv[l[5]-10+j][i]
                linesum += grayinv[l[5]-10+j][i]
            sumlist.append(linesum)
        blocksum = 0
        for s in range(len(sumlist)):
            if sumlist[s] > 3000 and sumlist[s] < 7000:
                    ###  3k, 6k value need to change with different input img
                for j in range(blocklen):
                    img[l[5]-10+j][s] = (255,0,0)
                notesumls=[]
                for n in range(15):##7 lines
                    ln = n // 2
                    rem = n%2
                    notesum = 0
                    if n == 14:
                        for h in range(blockwide):
                            for v in range(blockwide):
                                notesum += imgg[l[6]-blockwide+h][s+v]
                    elif n < 7:
                        for h in range(blockwide):
                            for v in range(blockwide):
                                notesum += imgg[l[ln]-(rem*blockwide//2)+h][s-v]
                                #img[l[ln]-(rem*blockwide//2)+h][s-v] = (0,255,0)
                    else:
                        for h in range(blockwide):
                            for v in range(blockwide):
                                notesum += imgg[l[ln]-(rem*blockwide//2)+h][s+v]
                                #img[l[ln]-(rem*blockwide//2)+h][s+v] = (0,255,0)
                    notesumls.append(notesum)
                note = notesumls.index(min(notesumls))
                notels.append(note)
                ####scan grayinv to find if two notes are connected
                if note >= 7:##connection on bot
                    if prevconnect == 1:
                        prevconnect = 0
                        sumconnect = 0
                        #sumconnect2 = 0
                        for i in range(blockwide*4):##l0 to l3
                            for j in range(blockwide*2):
                                #sumconnect += grayinv[l[3]+i][s+j]
                                sumconnect += grayinv[l[3]+i][s-j]
                                img[l[3]+i][s+j] = (0,255,0)
                        print(sumconnect)
                        if sumconnect > 12000:
                            connected2.append(len(notels)-1)
                        elif sumconnect > 7000:
                            connected.append(len(notels)-1)
                    else:
                        sumconnect = 0
                        for i in range(blockwide*4):##l0 to l3
                            for j in range(blockwide*2):
                                sumconnect += grayinv[l[3]+i][s+j]
                                img[l[3]+i][s+j] = (0,255,0)
                        if sumconnect > 8000:
                            connected.append(len(notels)-1)
                            prevconnect = 1
                if note < 7:##connection on top
                    sumconnect = 0
                    for i in range(blockwide*4):##l0 to l3
                        for j in range(blockwide*2):
                            sumconnect += grayinv[l[3]-i][s+j]
                            img[l[3]-i][s+j] = (0,255,0)
                    if sumconnect > 8000:
                        connected.append(len(notels)-1)
                            
    return [notels,connected,connected2]


if __name__ == '__main__':
    img = cv2.imread('song1.png')
    imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)###gray img for scaning
    grayinv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)##invsered gray img for finding if two notes are connnected
    for i in range(len(grayinv)):
        for j in range(len(grayinv[0])):
            if (grayinv[i][j] < 15):
                grayinv[i][j] = 255
            else:
                grayinv[i][j] = 0
    edges = cv2.Canny(img,50,150,apertureSize = 3)

    minLineLength=100
    header = 90
    diff = 10
    gap = 30
    blocklen = 50
    smallgap = 5
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

    sortlines = []
    for l in lines:
        sortlines.append(l[0])
        
    sortlines.sort(key=lambda x: x[1])
    linegroup = getlinesgroup(sortlines)
    print(linegroup)

    blockwide = linegroup[0][1] - linegroup[0][0]
    line7groups = line7groupsFun(linegroup)
    print(line7groups)
    notesres = getNotesAndTmpo(grayinv,imgg,line7groups,blockwide)
    notels = notesres[0]
    connected = notesres[1]
    connected2 = notesres[2]
    print(notels)
    print(connected)
    print(connected2)
    #cv2.imwrite('grayinv.jpg',grayinv)
    cv2.imwrite('houghlines5.jpg',img)
    cv2.imwrite('grayinv.jpg',grayinv)
    musicls = []
    for note in notels:
        musicls.append(mapnote(note))
    print(musicls)
    turnToMidi(musicls,connected)
