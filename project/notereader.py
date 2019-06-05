#------------------------------
# "Primitive" Sheet Music Reader
#------------------------------
import numpy as np
import cv2
from midiutil import MIDIFile
import math
import random
import os

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



def turnToMidi(musicls):
    midi = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    pitch = 0
    duration = 1
    volume = 100

    midi.addTrackName(track, time, "test")
    midi.addTempo(track, time, 60)
    music = []
    for i, pitch in enumerate(musicls):
        if pitch[1] == True:
            duration = 1 / 2
            midi.addNote(track, channel, pitch[0], time, duration, volume)
            time += 1 / 2
            music.append([pitch[0], 1 / 2])
        else:
            duration = 1
            midi.addNote(track, channel, pitch[0], time, duration, volume)
            time += 1
            music.append([pitch[0], 1])
    print(music)
    midi_file = open("mary.mid", 'wb')
    midi.writeFile(midi_file)
    midi_file.close()


def focus_staffs_pixel_accurate_binary(canny_img_array):
    height, width = canny_img_array.shape
    lines = []
    new_img_array = np.zeros((height, width), dtype=np.uint8)
    for row in range(height):
        if sum(canny_img_array[row, :])/255 >= 0.55 * width:
            lines.append(row) 


    counter = 0 #for keeping track of staffs
    flag = False #for skipping every other line in "lines" because the canny image draws 2 edges around every 1 edge in the original image.
    staffs = []
    staff_lines = []
    for idx in range(len(lines)):
        if flag is True:
            line_position = int(abs(lines[idx]+lines[idx-1])/2)
            new_img_array[line_position, :] = 255
            staff_lines.append(line_position)
            counter += 1
            if counter == 5:
                staffs.append(staff_lines[:])
                staff_lines.clear()
                counter = 0
            flag = False
        else:
            flag = True


    return new_img_array, staffs#, staff_line_jumps, X


def add_upper_lower_staff_lines(number_lines_to_add, staffs, line_gap):
    new_staffs = staffs
    for idx in range(len(new_staffs)):
        for n in range(0, number_lines_to_add):
            new_staffs[idx].append(new_staffs[idx][-1]+line_gap) #add lower staff line
            new_staffs[idx].insert(0, new_staffs[idx][0]-line_gap) #add upper staff line

    return new_staffs


def invgray(grayinv):
    for i in range(len(grayinv)):
        for j in range(len(grayinv[0])):
            if (grayinv[i][j] == 0):
                grayinv[i][j] = 255
            else:
                grayinv[i][j] = 0
        return grayinv



def getNotesAndTmpo_lite(staffs, gap_len, five_line_staff_height):
    notes_picked_up = []

    note_diameter = int(gap_len * 1.2) #approximate horizontal length of the note in relation to the length of the gap between staff lines.
    some_threshold = note_diameter*1.5 
    note_area = gap_len * note_diameter #approximate area of the note
    staff_height = abs(staffs[0][0] - staffs[0][-1])
    for staff in staffs:
        staff.sort(key=lambda x: x, reverse=True)
        print(staff)
        sumlist = []
        for col in range(len(inverse[0])): # iterate through the columns(horizontal scan)
            linesum = sum(inverse[staff[-1]:staff[-1]+staff_height, col])#sum(grayinv[l[0]:l[0]+staff_height, col])
            sumlist.append(linesum) #list of sums of total pixel values in a column of pixels
        previous_line_index = 0
        for s in range(len(sumlist)): #iterate through every column of pixels of the staff height horizontaly across a staff. Each s is a column position of a note along a row.
            #: #threshold indicating a note. #s-previous_line_index prevents registering the same note twice since the canny image gives 2 vertical edges per note.
            if sumlist[s] >= 255*five_line_staff_height*0.8 and s - previous_line_index >= some_threshold \
                    and eroded[staff[-1]:staff[-1]+staff_height, s].sum() < five_line_staff_height * 255*0.7:
                # Generate image data for neural network
                # note_im = img2[staff[-1]:staff[-1] + staff_height:, s - note_diameter:s + note_diameter]
                # note_im = cv2.resize(note_im, dsize=(10, 40))
                # cv2.imwrite('C:/Users/mike/Desktop/temp/note_images/note' + str(random.randint(1, 100000)) + '.jpg',
                #             note_im)
                for j in range(staff_height): #50 pixels row wise(vertical scan)
                    img[staff[-1]:staff[-1]+staff_height:, s] = (255, 0, 0) #draw a vertical line the height of the staff where the note is



                notes_list = []
                connection_detected_flag = False
                connection_line = 0
                #iterate through each line and line gap along each column of pixels the length of the height of the staff
                for idx, line in enumerate(staff):  # we always start at the bottom line gap. ---------------------------------------------------------------------------------------------------------------------
                    # The following is going to be used for finding the value of the notes.
                    # We scan verticaly on both the left and right side of where the tail of each note is since that is where
                    # the vertical note detection line is drawn. Some notes are on the left side of its tail and some notes
                    # are on the right side of its tail which is why we look at both sides.
                    notesum_gap_right = eroded[line: line + gap_len, s: s + note_diameter].sum()
                    notesum_gap_left = eroded[line: line + gap_len, s - note_diameter: s].sum()
                    notesum_line_right = eroded[int(math.ceil(line - gap_len / 2)): int(math.ceil(line + gap_len / 2)), s: s + note_diameter].sum()
                    notesum_line_left = eroded[int(math.ceil(line - gap_len / 2)): int(math.ceil(line + gap_len / 2)), s - note_diameter: s].sum()


                    note_set_current = [notesum_gap_right, notesum_gap_left, notesum_line_right, notesum_line_left]

                    # The following is going to be used for finding the tempo of the note as in if a note is connected to another note by a horizontal line.
                    # The difference in detecting the connection between lines from detecting just the note labels is that we scan verticaly
                    # both on the left and right side of where the tail of the note is by 2 times the diameter of the note.
                    line_offset = int(gap_len*0.5)
                    if connection_detected_flag is False:
                        for row_idx in range(int(gap_len+line_offset)):
                            if line - line_offset+row_idx < len(eroded):
                                connection_gap_right = eroded[line - line_offset + row_idx, s: s + note_diameter * 2].sum()
                                connection_gap_left = eroded[line - line_offset + row_idx, s - note_diameter * 2: s].sum()

                                if connection_gap_right >= note_diameter * 2 * 255:
                                    connection_detected_flag = True
                                    img[line: line + int(gap_len/2), s:s + note_diameter*2] = (255, 0, 255)
                                    connection_line = line
                                    break
                                if connection_gap_left >= note_diameter * 2 * 255:
                                    connection_detected_flag = True
                                    img[line: line + int(gap_len / 2), s - note_diameter * 2:s] = (255, 0, 255)
                                    connection_line = line
                                    break
                    if line != connection_line:
                        notes_list.append([line, note_set_current, idx])


                notes_list.sort(key=lambda x: max(x[1]))
                note_index = notes_list[-1][2]
                note_current = notes_list[-1][1]
                line = notes_list[-1][0]
                line_gap_index = note_current.index(max(note_current))
                if max(note_current) >= note_area * 255 * 0.3 and max(note_current) < note_area*255:
                    if line_gap_index == 0:
                        img[line: line + gap_len, s:s + note_diameter] = (0, 0, 255)  # gap right
                        notes_picked_up.append((note_index * 2, connection_detected_flag))
                    elif line_gap_index == 1:
                        img[line: line + gap_len, s - note_diameter:s] = (0, 255, 0)  # gap left
                        notes_picked_up.append((note_index * 2, connection_detected_flag))
                    elif line_gap_index == 2:
                        img[int(math.ceil(line - gap_len / 2)): int(math.ceil(line + gap_len / 2)),
                        s:s + note_diameter] = (0, 0, 255)  # line right
                        notes_picked_up.append((note_index * 2 + 1, connection_detected_flag))
                    else:
                        img[int(math.ceil(line - gap_len / 2)): int(math.ceil(line + gap_len / 2)),
                        s - note_diameter:s] = (0, 255, 0)  # line left
                        notes_picked_up.append((note_index * 2 + 1, connection_detected_flag))
                    previous_line_index = s

    return notes_picked_up


if __name__ == '__main__':
    song = 'mary.JPG'
    img = cv2.imread(song)
    img2 = cv2.imread(song, cv2.IMREAD_GRAYSCALE)
    imgg = cv2.imread(song, cv2.IMREAD_GRAYSCALE)  ###gray img for scaning

    inverse = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  ##invsered gray img for finding if two notes are connnected
    for i in range(len(inverse)):
        for j in range(len(inverse[0])):
            if (inverse[i][j] <=200):
                inverse[i][j] = 255
            else:
                inverse[i][j] = 0
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(inverse, kernel, iterations=1)
    eroded = cv2.dilate(eroded, kernel, iterations=1)
    #cv2.imshow('gray', eroded)
    cv2.imwrite('gray.jpg',eroded)
    #cv2.waitKey(0)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # Get location of staff lines
    lines, staffs = focus_staffs_pixel_accurate_binary(edges)
    print(staffs)
    staff_line_gap = staffs[0][1] - staffs[0][0]
    five_line_staff_height = staffs[0][-1] - staffs[0][0]

    # add extra staff lines
    staffs = add_upper_lower_staff_lines(3, staffs, staff_line_gap)

    notes = getNotesAndTmpo_lite(staffs, staff_line_gap, five_line_staff_height)
    #print('notes:', notes)
    #cv2.imshow('gray', img)
    cv2.imwrite('gray2.jpg',img)
    notels = []
    for note in notes:
        notels.append([mapnote(note[0]),note[1]])
    print('notels:', notels)       
    turnToMidi(notels)
    
    #cv2.waitKey()
