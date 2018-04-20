import cv2
from bidi.algorithm import get_display
import time
import numpy as np
from googletrans import Translator
import pytesseract
import arabic_reshaper
from PIL import Image, ImageDraw

def number_of_characters(filename):
    s = 'abcdefghijklmnopqrstuvwxyz'
    i = 0
    with open(filename) as f:
        for line in f:
            l = line.lower()
            for k in l:
                if (k in s):
                    i += 1
    return i
def count_string(s,filename):
    c=0
    with open(filename,'r',encoding="utf-8") as f:
        for line in f:
            l=line.lower()
            c+=l.count(s)
    return c
def uni_gram(filename):
    c=[]
    s='abcdefghijklmnopqrstuvwxyz'
    for ch in s:
        c.append(count_string(ch,filename))
    return c
def bie_gram(filename):
    c=[]
    s='abcdefghijklmnopqrstuvwxyz'
    for ch1 in s:
        k = []
        for ch2 in s:
            k.append(count_string(ch1+ch2,filename))
        c.append(k)

    return c
def tri_gram(filename):
    c = []
    s = 'abcdefghijklmnopqrstuvwxyz'
    for ch1 in s:
        k1 = []
        for ch2 in s:
            k2 = []
            for ch3 in s:
                k2.append(count_string(ch1 + ch2 + ch3, filename))
            k1.append(k2)
        c.append(k1)
    return c
def forth_gram(filename):
    c = []
    s = 'abcdefghijklmnopqrstuvwxyz'
    for ch1 in s:
        k1 = []
        for ch2 in s:
            k2 = []
            for ch3 in s:
                k3=[]
                for ch4 in s:
                    k3.append(count_string(ch1 + ch2 + ch3 + ch4, filename))
                k2.append(k3)
            k1.append(k2)
        c.append(k1)
    return c
def fifth_gram(filename):
    c = []
    s = 'abcdefghijklmnopqrstuvwxyz'
    for ch1 in s:
        k1 = []
        for ch2 in s:
            k2 = []
            for ch3 in s:
                k3=[]
                for ch4 in s:
                    k4=[]
                    for ch5 in s:
                        k4.append(count_string(ch1 + ch2 + ch3 + ch4+ch5, filename))
                    k3.append(k4)
                k2.append(k3)
            k1.append(k2)
        c.append(k1)
    return c

def get_probablities_bie(filename):
    c1=uni_gram(filename)
    c2=bie_gram(filename)
    result=[]
    for i,ii in zip(c2,c1):
        temp=[]
        for j in i:
            if(ii>0):
                temp.append(j/ii)
            else:
                temp.append(0)
        result.append(temp)
    return result
def get_probablities_tri(filename):
    c2 = bie_gram(filename)
    c3 = tri_gram(filename)
    prob_bie=get_probablities_bie(filename)
    result=[]
    for i,i1,p in zip(c2,c3,prob_bie):
        temp1=[]
        for j,j1,p1 in zip(i,i1,p):
            temp = []
            for k in j1:
                if(j>0):
                    temp.append((k/j))
                else:
                    temp.append(0)
            temp1.append(temp)
        result.append(temp1)
    return result
def get_probablities_forth(filename):
    c3 = tri_gram(filename)
    c4=forth_gram(filename)
    result=[]
    for i,i1 in zip(c3,c4):
        temp1=[]
        for j,j1 in zip(i,i1):
            temp = []
            for jj,j2 in zip(j,j1):
                temp2 = []
                for k in j2:
                    if(jj>0):
                        temp2.append((k/jj))
                    else:
                        temp2.append(0)
                temp.append(temp2)
            temp1.append(temp)
        result.append(temp1)
    return result
def get_probablities_fifth(filename):
    c4 = forth_gram(filename)
    c5 = fifth_gram(filename)
    result=[]
    for i,i1 in zip(c4,c5):
        temp1=[]
        for j,j1 in zip(i,i1):
            temp = []
            for jj,j2 in zip(j,j1):
                temp2 = []
                for jjj, j3 in zip(jj, j2):
                    temp3=[]
                    for k in j3:
                        if(jjj>0):
                            temp3.append((k/jjj))
                        else:
                            temp3.append(0)
                    temp2.append(temp3)
                temp.append(temp2)
            temp1.append(temp)
        result.append(temp1)
    return result

def LM_bie():
    filename = "trainEng.txt"
    c1=get_probablities_bie(filename)
    filename = "trainGer.txt"
    c2 = get_probablities_bie(filename)
    filename = "trainFre.txt"
    c3 = get_probablities_bie(filename)
    s = 'abcdefghijklmnopqrstuvwxyz'
    file = open("testfileEng_bie.txt", "w")
    file1 = open("testfileGer_bie.txt", "w")
    file2 = open("testfileFre_bie.txt", "w")
    for i, j,k,l in zip(s, c1,c2,c3):
        for i1, j1,k1,l1 in zip(s, j,k,l):
            file.write(i + i1  + " " + str(j1) + ',')
            file1.write(i + i1 + " " + str(k1) + ',')
            file2.write(i + i1 + " " + str(l1) + ',')

    file.close()
    file1.close()
    file2.close()

    return c1,c2,c3
def LM_tri():
    filename = "trainEng.txt"
    c1=get_probablities_tri(filename)
    filename = "trainGer.txt"
    c2 = get_probablities_tri(filename)
    filename = "trainFre.txt"
    c3 = get_probablities_tri(filename)

    s = 'abcdefghijklmnopqrstuvwxyz'
    file = open("testfileEng_tri.txt", "w")
    file1 = open("testfileGer_tri.txt", "w")
    file2 = open("testfileFre_tri.txt", "w")
    for i,j,k,l in zip(s,c1,c2,c3):
        for i1, j1,k1,l1 in zip(s, j,k,l):
            for i2, j2,k2,l2 in zip(s, j1,k1,l1):
                file.write(i + i1 + i2 + " " + str(j2) + ',')
                file1.write(i + i1 + i2 + " " + str(k2) + ',')
                file2.write(i + i1 + i2 + " " + str(l2) + ',')
    file.close()
    file1.close()
    file2.close()

    return c1,c2,c3
def LM_forth():
    filename = "trainEng.txt"
    c1=get_probablities_forth(filename)
    filename = "trainGer.txt"
    c2 = get_probablities_forth(filename)
    filename = "trainFre.txt"
    c3 = get_probablities_forth(filename)

    s = 'abcdefghijklmnopqrstuvwxyz'
    file = open("testfileEng_forth.txt", "w")
    file1 = open("testfileGer_forth.txt", "w")
    file2 = open("testfileFre_forth.txt", "w")
    for i,j,k,l in zip(s,c1,c2,c3):
        for i1, j1,k1,l1 in zip(s, j,k,l):
            for i2, j2,k2,l2 in zip(s, j1,k1,l1):
                for i3,j3,k3,l3 in zip(s,j2,k2,l2):
                    file.write(i + i1 + i2 + i3 + " " + str(j3) + ',')
                    file1.write(i + i1 + i2 + i3 + " " + str(k3) + ',')
                    file2.write(i + i1 + i2 + i3 + " " + str(l3) + ',')
    file.close()
    file1.close()
    file2.close()
    return c1,c2,c3
def LM_fifth():
    filename = "trainEng.txt"
    c1=get_probablities_fifth(filename)
    filename = "trainGer.txt"
    c2 = get_probablities_fifth(filename)
    filename = "trainFre.txt"
    c3 = get_probablities_fifth(filename)

    s = 'abcdefghijklmnopqrstuvwxyz'
    file = open("testfileEng_fifth.txt", "w")
    file1 = open("testfileGer_fifth.txt", "w")
    file2 = open("testfileFre_fifth.txt", "w")
    for i,j,k,l in zip(s,c1,c2,c3):
        for i1, j1,k1,l1 in zip(s, j,k,l):
            for i2, j2,k2,l2 in zip(s, j1,k1,l1):
                for i3,j3,k3,l3 in zip(s,j2,k2,l2):
                    for i4, j4,k4,l4 in zip(s, j3,k3,l3):
                        file.write(i + i1 + i2 + i3 + i4 + " " + str(j4) + ',')
                        file1.write(i + i1 + i2 + i3 + i4 + " " + str(k4) + ',')
                        file2.write(i + i1 + i2 + i3 + i4 + " " + str(l4) + ',')
    file.close()
    file1.close()
    file2.close()
    return c1,c2,c3

def check_strange_char(filename):
    s="ßöä"
    s1="èçàùéëïâêîôû"
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            for l in line:
                a=l.lower()
                if(a in s):
                    return "ger"
                elif(a in s1):
                    return "fre"
    return "eng"

def train_bie(filename):
    temp1,temp2,temp3=LM_bie()
    test=bie_gram(filename)
    eng=0
    ger=0
    fre=0
    for i,a,b,c in zip(test,temp1,temp2,temp3):
        for i1,a1,b1,c1 in zip(i,a,b,c):
            if(i1>0):
                eng+=a1
                ger+=b1
                fre+=c1
    if(eng>ger and eng>fre):
        return "eng"
    elif(ger>eng and ger>fre):
        return "ger"
    elif (fre > eng and ger < fre):
        return "fre"
    else:
        return "number"
def detect_bie_tested(filename):
    check = check_strange_char("test.txt")
    if (check != "eng"):
        return check
    testEng_bie = open("testfileEng_bie.txt", "r")
    testGer_bie = open("testfileGer_bie.txt", "r")
    testFre_bie = open("testfileFre_bie.txt", "r")
    testedEng = testEng_bie.read()
    testedGer = testGer_bie.read()
    testedFre = testFre_bie.read()
    s = "abcdefghijklmnopqrstuvwxyz"
    eng=0
    ger=0
    fre=0
    with open(filename,'r',encoding="utf-8") as f:
        for line in f:
            l = line.lower()
            i=0
            while i<len(l)-1:
                a = l[i]
                b = l[i + 1]
                i+=1
                p=0
                if(a in s and b in s ):
                    p=1
                if p==0:
                    continue
                k1 = ''
                k2 = ''
                k3 = ''
                o=3
                while(testedEng[testedEng.find(a+b).__pos__() + o]!=','):
                    k1+=testedEng[testedEng.find(a+b).__pos__() + o]
                    o+=1
                o=3
                while (testedGer[testedGer.find(a + b).__pos__() + o] != ','):
                    k2+= testedGer[testedGer.find(a + b ).__pos__() + o]
                    o += 1
                o=3
                while (testedFre[testedFre.find(a + b ).__pos__() + o] != ','):
                    k3+= testedFre[testedFre.find(a + b).__pos__() + o]
                    o+= 1
                eng+=float(k1)
                ger+=float(k2)
                fre+=float(k3)
    if(eng>ger and eng>fre):
        return "eng"
    elif(ger>eng and ger>fre):
        return "ger"
    elif (fre > eng and ger < fre):
        return "fre"
    else:
        return "eng"
def train_tri(filename):
    temp1,temp2,temp3=LM_tri()
    test = tri_gram(filename)
    eng=0
    ger=0
    fre=0
    for i, a, b, c in zip(test, temp1, temp2, temp3):
        for i1, a1, b1, c1 in zip(i, a, b, c):
            for i2, a2, b2, c2 in zip(i1, a1, b1, c1):
                if (i2 > 0):
                    eng += a2
                    ger += b2
                    fre += c2
    if(eng>ger and eng>fre):
        return "eng"
    elif(ger>eng and ger>fre):
        return "ger"
    elif (fre > eng and ger < fre):
        return "fre"
    else:
        return "number"
def detect_tri_tested(filename):
    check = check_strange_char("test.txt")
    testEng_tri = open("testfileEng_tri.txt", "r")
    testGer_tri = open("testfileGer_tri.txt", "r")
    testFre_tri = open("testfileFre_tri.txt", "r")
    testedEng = testEng_tri.read()
    testedGer = testGer_tri.read()
    testedFre = testFre_tri.read()
    if (check != "eng"):
        return check
    s = "abcdefghijklmnopqrstuvwxyz"
    eng = 0
    ger = 0
    fre = 0

    with open(filename,'r',encoding="utf-8") as f:
        for line in f:
            l = line.lower()
            i=0
            while i<len(l)-2:
                a = l[i]
                b = l[i + 1]
                c = l[i + 2]
                i+=1
                p=0
                if(a in s and b in s and c in s ):
                    p=1
                if p==0:
                    continue
                k1 = ''
                k2 = ''
                k3 = ''
                o=4
                while(testedEng[testedEng.find(a+b+c).__pos__() + o]!=','):
                    k1+=testedEng[testedEng.find(a+b+c).__pos__() + o]
                    o+=1
                o=4
                while (testedGer[testedGer.find(a + b + c).__pos__() + o] != ','):
                    k2+= testedGer[testedGer.find(a + b + c).__pos__() + o]
                    o += 1
                o=4
                while (testedFre[testedFre.find(a + b + c).__pos__() + o] != ','):
                    k3+= testedFre[testedFre.find(a + b + c).__pos__() + o]
                    o+= 1
                eng+=float(k1)
                ger+=float(k2)
                fre+=float(k3)
    if(eng>ger and eng>fre):
        return "eng"
    elif(ger>eng and ger>fre):
        return "ger"
    elif (fre > eng and ger < fre):
        return "fre"
    else:
        return "eng"
def train_forth(filename):
    temp1,temp2,temp3=LM_forth()
    eng=0
    ger=0
    fre=0
    test = forth_gram(filename)
    for i, a, b, c in zip(test, temp1, temp2, temp3):
        for i1, a1, b1, c1 in zip(i, a, b, c):
            for i2, a2, b2, c2 in zip(i1, a1, b1, c1):
                for i3, a3, b3, c3 in zip(i2, a2, b2, c2):
                    if (i3 > 0):
                        eng += a3
                        ger += b3
                        fre += c3
    if(eng>ger and eng>fre):
        return "eng"
    elif(ger>eng and ger>fre):
        return "ger"
    elif (fre > eng and ger < fre):
        return "fre"
    else:
        return "number"
def detect_forth_tested(filename):
    check = check_strange_char("test.txt")

    testEng_forth = open("testfileEng_forth.txt", "r")
    testGer_forth = open("testfileGer_forth.txt", "r")
    testFre_forth = open("testfileFre_forth.txt", "r")
    testedEng = testEng_forth.read()
    testedGer = testGer_forth.read()
    testedFre = testFre_forth.read()
    testEng_tri = open("testfileEng_tri.txt", "r")
    testGer_tri = open("testfileGer_tri.txt", "r")
    testFre_tri = open("testfileFre_tri.txt", "r")
    testEng_tri = testEng_tri.read()
    testGer_tri = testGer_tri.read()
    testFre_tri = testFre_tri.read()
    testEng_bie = open("testfileEng_bie.txt", "r")
    testGer_bie = open("testfileGer_bie.txt", "r")
    testFre_bie = open("testfileFre_bie.txt", "r")
    testEng_bie = testEng_bie.read()
    testGer_bie = testGer_bie.read()
    testFre_bie = testFre_bie.read()

    if (check != "eng"):
        return check
    s = "abcdefghijklmnopqrstuvwxyz"
    eng = 0
    ger = 0
    fre = 0
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            try:
                f.decode('utf-8')
            except:
                f=f
            l = line.lower()
            i=0
            while i<len(l)-1:
                a = l[i]
                b = l[i + 1]
                p=0
                if(a in s and b in s ):
                    p=1
                if p==0:
                    i+=1
                    continue
                k1 = ''
                k2 = ''
                k3 = ''
                o=3
                while(testEng_bie[testEng_bie.find(a+b).__pos__() + o]!=','):
                    k1+=testEng_bie[testEng_bie.find(a+b).__pos__() + o]
                    o+=1
                o=3
                while (testGer_bie[testGer_bie.find(a + b).__pos__() + o] != ','):
                    k2+= testGer_bie[testGer_bie.find(a + b ).__pos__() + o]
                    o += 1
                o=3
                while (testFre_bie[testFre_bie.find(a + b ).__pos__() + o] != ','):
                    k3+= testFre_bie[testFre_bie.find(a + b).__pos__() + o]
                    o+= 1

                if (float(k1) > 0 and float(k2) == 0 and float(k3) == 0):
                    return "eng"
                elif (float(k1) == 0 and float(k2) > 0 and float(k3) == 0):
                    return "ger"
                elif (float(k1) == 0 and float(k2) == 0 and float(k3) > 0):
                    return "fre"
                if (float(k1) == 0 ):
                    k1= -5
                if (float(k2) == 0):
                    k2=-5
                if (float(k3) == 0):
                    k3=-5
                eng+=float(k1)
                ger+=float(k2)
                fre+=float(k3)
                if(i+2>len(l)-1):
                    i+=1
                    continue
                a = l[i]
                b = l[i + 1]
                c = l[i + 2]
                p = 0
                if (a in s and b in s and c in s):
                    p = 1
                if p == 0:
                    i+=1
                    continue
                k1 = ''
                k2 = ''
                k3 = ''
                o = 4
                while (testEng_tri[testEng_tri.find(a + b + c).__pos__() + o] != ','):
                    k1 += testEng_tri[testEng_tri.find(a + b + c).__pos__() + o]
                    o += 1
                o = 4
                while (testGer_tri[testGer_tri.find(a + b + c).__pos__() + o] != ','):
                    k2 += testGer_tri[testGer_tri.find(a + b + c).__pos__() + o]
                    o += 1
                o = 4
                while (testFre_tri[testFre_tri.find(a + b + c).__pos__() + o] != ','):
                    k3 += testFre_tri[testFre_tri.find(a + b + c).__pos__() + o]
                    o += 1
                if (float(k1) > 0 and float(k2) == 0 and float(k3) == 0):
                    return "eng"
                elif (float(k1) == 0 and float(k2) > 0 and float(k3) == 0):
                    return "ger"
                elif (float(k1) == 0 and float(k2) == 0 and float(k3) > 0):
                    return "fre"
                if (float(k1) == 0 ):
                    k1= -5
                if (float(k2) == 0):
                    k2=-5
                if (float(k3) == 0):
                    k3=-5
                eng += float(k1)
                ger += float(k2)
                fre += float(k3)
                if (i + 3 > len(l) - 1):
                    i+=1
                    continue
                a = l[i]
                b = l[i + 1]
                c = l[i + 2]
                d = l[i + 3]
                p = 0
                if (a in s and b in s and c in s and d in s):
                    p = 1
                if p == 0:
                    i+=1
                    continue
                k1 = ''
                k2 = ''
                k3 = ''
                o = 5

                while (testedEng[testedEng.find(a + b + c + d).__pos__() + o] != ','):
                    k1 += testedEng[testedEng.find(a + b + c + d).__pos__() + o]
                    o += 1
                o = 5
                while (testedGer[testedGer.find(a + b + c + d).__pos__() + o] != ','):
                    k2 += testedGer[testedGer.find(a + b + c + d).__pos__() + o]
                    o += 1
                o = 5
                while (testedFre[testedFre.find(a + b + c + d).__pos__() + o] != ','):
                    k3 += testedFre[testedFre.find(a + b + c + d).__pos__() + o]
                    o += 1
                if (float(k1) > 0 and float(k2) == 0 and float(k3) == 0):
                    return "eng"
                elif (float(k1) == 0 and float(k2) > 0 and float(k3) == 0):
                    return "ger"
                elif (float(k1) == 0 and float(k2) == 0 and float(k3) > 0):
                    return "fre"
                if (float(k1) == 0 ):
                    k1= -5
                if (float(k2) == 0):
                    k2=-5
                if (float(k3) == 0):
                    k3=-5
                eng += float(k1)
                ger += float(k2)
                fre += float(k3)
                i+=1
    if (eng > ger and eng > fre):
        return "eng"
    elif (ger > eng and ger > fre):
        return "ger"
    elif (fre > eng and ger < fre):
        return "fre"
    else:
        return "eng"
def train_fifth(filename):
    temp1,temp2,temp3=LM_fifth()
    eng=0
    ger=0
    fre=0
    test = fifth_gram(filename)
    for i, a, b, c in zip(test, temp1, temp2, temp3):
        for i1, a1, b1, c1 in zip(i, a, b, c):
            for i2, a2, b2, c2 in zip(i1, a1, b1, c1):
                for i3, a3, b3, c3 in zip(i2, a2, b2, c2):
                    for i4, a4, b4, c4 in zip(i3, a3, b3, c3):
                        if (i4 > 0):
                            eng += a4
                            ger += b4
                            fre += c4
    if(eng>ger and eng>fre):
        return "eng"
    elif(ger>eng and ger>fre):
        return "ger"
    elif (fre > eng and ger < fre):
        return "fre"
    else:
        return "number"
def detect_fifth_tested(filename):
    check = check_strange_char("test.txt")
    testEng_fifth = open("testfileEng_fifth.txt", "r")
    testGer_fifth = open("testfileGer_fifth.txt", "r")
    testFre_fifth = open("testfileFre_fifth.txt", "r")
    testedEng = testEng_fifth.read()
    testedGer = testGer_fifth.read()
    testedFre = testFre_fifth.read()
    testEng_tri = open("testfileEng_tri.txt", "r")
    testGer_tri = open("testfileGer_tri.txt", "r")
    testFre_tri = open("testfileFre_tri.txt", "r")
    testEng_tri = testEng_tri.read()
    testGer_tri = testGer_tri.read()
    testFre_tri = testFre_tri.read()
    testEng_bie = open("testfileEng_bie.txt", "r")
    testGer_bie = open("testfileGer_bie.txt", "r")
    testFre_bie = open("testfileFre_bie.txt", "r")
    testEng_bie = testEng_bie.read()
    testGer_bie = testGer_bie.read()
    testFre_bie = testFre_bie.read()
    testEng_forth = open("testfileEng_forth.txt", "r")
    testGer_forth = open("testfileGer_forth.txt", "r")
    testFre_forth = open("testfileFre_forth.txt", "r")
    testEng_forth = testEng_forth.read()
    testGer_forth = testGer_forth.read()
    testFre_forth = testFre_forth.read()

    if (check != "eng"):
        return check
    s = "abcdefghijklmnopqrstuvwxyz"
    eng = 0
    ger = 0
    fre = 0
    with open(filename,'r',encoding="utf-8") as f:
        for line in f:
            l = line.lower()
            i=0
            while i<len(l)-1:
                a = l[i]
                b = l[i + 1]
                p=0
                if(a in s and b in s ):
                    p=1
                if p==0:
                    i+=1
                    continue
                k1 = ''
                k2 = ''
                k3 = ''
                o=3
                while(testEng_bie[testEng_bie.find(a+b).__pos__() + o]!=','):
                    k1+=testEng_bie[testEng_bie.find(a+b).__pos__() + o]
                    o+=1
                o=3
                while (testGer_bie[testGer_bie.find(a + b).__pos__() + o] != ','):
                    k2+= testGer_bie[testGer_bie.find(a + b ).__pos__() + o]
                    o += 1
                o=3
                while (testFre_bie[testFre_bie.find(a + b ).__pos__() + o] != ','):
                    k3+= testFre_bie[testFre_bie.find(a + b).__pos__() + o]
                    o+= 1

                if (float(k1) > 0 and float(k2) == 0 and float(k3) == 0):
                    return "eng"
                elif (float(k1) == 0 and float(k2) > 0 and float(k3) == 0):
                    return "ger"
                elif (float(k1) == 0 and float(k2) == 0 and float(k3) > 0):
                    return "fre"
                if (float(k1) == 0 ):
                    k1= -5
                if (float(k2) == 0):
                    k2=-5
                if (float(k3) == 0):
                    k3=-5
                eng+=float(k1)
                ger+=float(k2)
                fre+=float(k3)
                if(i+2>len(l)-1):
                    i+=1
                    continue
                a = l[i]
                b = l[i + 1]
                c = l[i + 2]
                p = 0
                if (a in s and b in s and c in s):
                    p = 1
                if p == 0:
                    i+=1
                    continue
                k1 = ''
                k2 = ''
                k3 = ''
                o = 4
                while (testEng_tri[testEng_tri.find(a + b + c).__pos__() + o] != ','):
                    k1 += testEng_tri[testEng_tri.find(a + b + c).__pos__() + o]
                    o += 1
                o = 4
                while (testGer_tri[testGer_tri.find(a + b + c).__pos__() + o] != ','):
                    k2 += testGer_tri[testGer_tri.find(a + b + c).__pos__() + o]
                    o += 1
                o = 4
                while (testFre_tri[testFre_tri.find(a + b + c).__pos__() + o] != ','):
                    k3 += testFre_tri[testFre_tri.find(a + b + c).__pos__() + o]
                    o += 1
                if (float(k1) > 0 and float(k2) == 0 and float(k3) == 0):
                    return "eng"
                elif (float(k1) == 0 and float(k2) > 0 and float(k3) == 0):
                    return "ger"
                elif (float(k1) == 0 and float(k2) == 0 and float(k3) > 0):
                    return "fre"
                if (float(k1) == 0 ):
                    k1= -5
                if (float(k2) == 0):
                    k2=-5
                if (float(k3) == 0):
                    k3=-5
                eng += float(k1)
                ger += float(k2)
                fre += float(k3)
                if (i + 3 > len(l) - 1):
                    i+=1
                    continue
                a = l[i]
                b = l[i + 1]
                c = l[i + 2]
                d = l[i + 3]
                p = 0
                if (a in s and b in s and c in s and d in s):
                    p = 1
                if p == 0:
                    i+=1
                    continue
                k1 = ''
                k2 = ''
                k3 = ''
                o = 5

                while (testEng_forth[testEng_forth.find(a + b + c + d).__pos__() + o] != ','):
                    k1 += testEng_forth[testEng_forth.find(a + b + c + d).__pos__() + o]
                    o += 1
                o = 5
                while (testGer_forth[testGer_forth.find(a + b + c + d).__pos__() + o] != ','):
                    k2 += testGer_forth[testGer_forth.find(a + b + c + d).__pos__() + o]
                    o += 1
                o = 5
                while (testFre_forth[testFre_forth.find(a + b + c + d).__pos__() + o] != ','):
                    k3 += testFre_forth[testFre_forth.find(a + b + c + d).__pos__() + o]
                    o += 1
                if (float(k1) > 0 and float(k2) == 0 and float(k3) == 0):
                    return "eng"
                elif (float(k1) == 0 and float(k2) > 0 and float(k3) == 0):
                    return "ger"
                elif (float(k1) == 0 and float(k2) == 0 and float(k3) > 0):
                    return "fre"
                if (float(k1) == 0 ):
                    k1= -5
                if (float(k2) == 0):
                    k2=-5
                if (float(k3) == 0):
                    k3=-5
                eng += float(k1)
                ger += float(k2)
                fre += float(k3)
                if (i + 4 > len(l) - 1):
                    i+=1
                    continue
                a = l[i]
                b = l[i + 1]
                c = l[i + 2]
                d = l[i + 3]
                e = l[i + 4]
                p = 0
                if (a in s and b in s and c in s and d in s and e in s):
                    p = 1
                if p == 0:
                    i+=1
                    continue
                k1 = ''
                k2 = ''
                k3 = ''
                o = 6

                while (testedEng[testedEng.find(a + b + c + d + e).__pos__() + o] != ','):
                    k1 += testedEng[testedEng.find(a + b + c + d + e).__pos__() + o]
                    o += 1
                o = 6
                while (testedGer[testedGer.find(a + b + c + d + e).__pos__() + o] != ','):
                    k2 += testedGer[testedGer.find(a + b + c + d + e).__pos__() + o]
                    o += 1
                o = 6
                while (testedFre[testedFre.find(a + b + c + d + e).__pos__() + o] != ','):
                    k3 += testedFre[testedFre.find(a + b + c + d + e).__pos__() + o]
                    o += 1
                if (float(k1) > 0 and float(k2) == 0 and float(k3) == 0):
                    return "eng"
                elif (float(k1) == 0 and float(k2) > 0 and float(k3) == 0):
                    return "ger"
                elif (float(k1) == 0 and float(k2) == 0 and float(k3) > 0):
                    return "fre"
                if (float(k1) == 0 ):
                    k1= -5
                if (float(k2) == 0):
                    k2=-5
                if (float(k3) == 0):
                    k3=-5
                eng += float(k1)
                ger += float(k2)
                fre += float(k3)
                i+=1

    if(eng>ger and eng>fre):
        return "eng"
    elif(ger>eng and ger>fre):
        return "ger"
    elif (fre > eng and ger < fre):
        return "fre"
    else:
        return "eng"

def get_Acurccy(check,filename):
    error=0
    true=0
    with open("Get_Accuracy.txt") as f:
        for line in f:
            inp = ""
            outp=""
            temp=0
            for l in line:
                ch=l.lower()
                if(ch!=','and temp==0):
                    inp+=ch
                elif(ch==','):
                    temp=1
                elif (ch == '\n'):
                    continue
                else:
                    outp+=ch
            file = open(filename, "w")
            pass
            file.write(inp)
            file.close()
            if(check=='tri'):
                test=detect_tri_tested(filename)
            elif (check == 'forth'):
                test = detect_forth_tested(filename)
            elif (check == 'fifth'):
                test = detect_fifth_tested(filename)
            else:
                test = detect_bie_tested(filename)
            if(test==outp):
                true+=1
            else:
                error+=1
    return (true/(error+true))*100

    return c
def Processing_Image(Image,destination):
    test = Image
    large = cv2.imread(test)
    large = cv2.resize(large, (800, 850))
    large = cv2.fastNlMeansDenoisingColored(large, None, 10, 10, 7, 21)
    rgb = cv2.pyrDown(large)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    k = small
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 2))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    _, contours, _ = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)

    translator = Translator()
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y + h, x:x + w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
        if r > 0.45 and w > 20 and h > 10:
            temp = k[y:y + 3 + h, x:x + 3 + w]
            #_, bw = cv2.threshold(temp, 0.0, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)
            #cv2.imshow('rects', bw)
            #cv2.waitKey(0)

            text = pytesseract.image_to_string(temp, lang='eng+deu')
            if (text != ''):
                #rgb[y:y + 3 + h, x:x + 3 + w]=temp[5,5]
                print("The text is: ", text)
                file = open("test.txt", "w", encoding="utf-8")
                pass
                file.write(str(text))
                file.close()
                c = detect_forth_tested("test.txt")
                if(c=="eng"):
                    c='English'
                elif(c=="fre"):
                    c="French"
                else:
                    c="German"

                cv2.rectangle(rgb, (x - 3, y), (x + w, y + h), (0, 255, 0), 1)
                print('the language is: ', c)
                try:
                    t = translator.translate(text, dest=destination, src=c)

                except:
                    print("Translation Error ... Check the internet connection")
                    exit()
                print('the translation is: ', t.text)
                h1=t.text
                h1=h1.replace('\n',' ')
                d = ImageDraw.Draw(rgb)
                reshaped_text = arabic_reshaper.reshape(t.text)
                bidi_text = get_display(reshaped_text)
                font = cv2.FONT_HERSHEY_SIMPLEX
                if(temp[0,0]>=240):
                    rgb[y:y + 3 + h, x:x + 3 + w] = temp[0, 0]
                    #cv2.putText(rgb, bidi_text, (x, y +13), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                    d.text((x, y + 13), bidi_text, font=font, fill=(0, 255, 255, 255))

                else:
                    #cv2.putText(rgb, bidi_text, (x, y -3), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    d.text((x, y + 13), bidi_text, font=font, fill=(0, 255, 255, 255))

            else:
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), 1)

    if (float(time.time() - start_time) >= 60):
        print("%.2f minutes" % (float(time.time() - start_time) / 60))
    else:
        print("%.2f seconds" % (time.time() - start_time))

    cv2.imshow('rects', rgb)
    cv2.waitKey(0)


if __name__ == '__main__':
    start_time = time.time()

    #c=get_Acurccy('forth',"test.txt")
    #c=train_fifth("test.txt")
    #file = open("test.txt", "w")
    #pass
    #file.write("hello")
    #file.close()
    #c=detect_forth_tested("test.txt")

    #print(c)
    Processing_Image('25.jpg','Arabic')




