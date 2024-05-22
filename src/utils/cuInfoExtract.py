import os

path="/home/machado/Desktop/slabref37_info.html"
sum=0
modeProp=0
noMode=0
listaModos = [0]*35

with open(path) as html:
    for line in html.readlines():
        for word in line.split(" "):
            if "Intra_Ang" in word:
                sum +=1
                if "(18" in word.split(")"):
                    modeProp+=1
                    #print(word)
                    listaModos[18]+=1
                else:
                    mode = int(word.split(")")[0].split("(")[1])
                    #
                    if mode == 35: 
                        print(mode)
                    listaModos[mode]+=1
                    noMode+=1
                    #sum+=1
            elif "Intra_Planar" in word:
                #print(word)
                listaModos[0]+=1
                noMode+=1
                sum+=1
            elif "Intra_DC" in word:
                listaModos[1]+=1
                noMode+=1
                sum+=1

      

for i, count in enumerate(listaModos):
    print( count/sum)
print(sum)
print(noMode)
print(modeProp)
print("%Prop", modeProp/sum)
print("%ref", noMode/sum)