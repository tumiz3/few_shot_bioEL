import json

name="bianxie"
outputPath="./"+name+"keys.json"
with open(name+"_keys_infromation.txt","r") as f:
    keyInformation=f.readlines()



urlInformation=keyInformation[0].strip()
keysText=keyInformation[1:]

url=urlInformation.split(" ")[-1]
keys=[]
for line in keysText:
    key=line.split(" ")[-1].strip()
    keys.append(key)

allInformation={"baseUrl":url,"keys":keys}
with open(outputPath,"w") as f:
    json.dump(allInformation,f)
