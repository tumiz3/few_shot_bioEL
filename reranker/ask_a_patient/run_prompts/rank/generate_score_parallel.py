import json
import multiprocessing
import numpy as np
import os
import re
import argparse
from tqdm import tqdm
from copy import deepcopy
from openai import OpenAI

def load_keys(keyPath:str) -> list:
    with open(keyPath,"r") as f:
        keys=json.load(f)
    return keys

def load_data(testDataPath:str) -> dict:
    with open(testDataPath,"r") as f:
        testData=json.load(f)
    return testData

def split_data(data:list)->list:
    split_list = np.array_split(data, 40)
    split_list = [arr.tolist() for arr in split_list]
    return split_list

def load_dict(entityKbPath:str) -> dict:
    with open(entityKbPath,"r") as f:
        entityKb=json.load(f)
    return entityKb

def load_prompt(promptPath:str,shot_number:int) -> str:
    #获取指定位置的prompt，同时根据shot_number选择prompt中例子的数量
    with open(promptPath,"r") as f:
        prompt=f.read()
    
    prompts=prompt.split("Example")
    if "mix" in prompt:
        shot_number=2*shot_number
    if "all" in prompt:
        shot_number=3*shot_number
    usedPrompt=prompts[:shot_number+1]
        
    newPrompt="Example".join(usedPrompt).strip()

    return newPrompt

def process_test_data(test:dict) -> list:
    testData=[]
    for item in test["queries"]:
        mention=item["mentions"][0]["mention"]
        candidatesData=item["mentions"][0]["candidates"]
        golden_cui=item["mentions"][0]["golden_cui"]
        candidates=[candidate["name"] for candidate in candidatesData]
        concepts=[candidate["labelcui"] for candidate in candidatesData]
        labels=[candidate["label"] for candidate in candidatesData]
        testData.append({"mention":mention,"candidates":candidates,"labels":labels,"golden_cui":golden_cui,"concepts":concepts})
    return testData
# def is_float(s):
#         try:
#             float(s)
#             return True
#         except ValueError:
#             return False

# def contains_float(string):
#     # 正则表达式匹配浮点数，包括整数、小数、带符号的数
#     pattern = r'[-+]?\d*\.\d+|\d+'
#     return bool(re.search(pattern, string))

# def extract_floats(string):
#     # 正则表达式匹配浮点数，包括整数、小数、带符号的数
#     pattern = r'[-+]?\d*\.\d+|\d+'
#     return [match for match in re.findall(pattern, string)][0] 

def create_client(base_url:str,api_key:str):
    client = OpenAI(base_url=base_url,api_key=api_key)
    return client

def ask(prompt,client,temperature,max_tokens):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",#Currently points to gpt-3.5-turbo-0125,input max tokens:16385,output max tokens:4096
        messages=[
            {"role":"system","content":"You are a medical expert, you know exactly the meanings of various medical terms. You can capture the similarities and differences between two medical terms and determine whether these two medical terms are different names for the same medical entity."},
            {"role":"user","content":prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    ans = response.choices[0].message.content
    return ans

def run_prompt_with_score(manual_prompt:str,testData:list,base_url,key,temperature,max_tokens,intermediatePath):
    client=create_client(base_url,key)
    GeneratedAnswers={}
    generatedOutput={}

    for sample in tqdm(testData):
        mention=sample["mention"]
        candidates=sample["candidates"]
        GeneratedAnswers[mention]=[]
        # for i in range(len(candidates)):

        i=0
        generatedOutput[mention]=[]
        while i<len(candidates):
            candidate=candidates[i]
            prompt=manual_prompt       
            prompt=prompt.replace("mention",mention)
            prompt=prompt.replace("unentity",candidate)
            try:
                ans=ask(prompt,client,temperature,64)
            except:
                continue
            generatedOutput[mention].append(ans)
            # ans=ask(prompt,client,temperature,64)
            i+=1
            # print(prompt)
            # print("\n")
            # print(ans)
            # print("\n")
            # print("-"*60)
            # # if ":" in ans:
            #     ans=ans.split(":")[1].strip(" \t,.，。\n")
            # if not is_float(ans):
            #         if contains_float(ans):
            #             ans=extract_floats(ans)
            #         else:
            #             ans="0.0"
            
            # print("\n")
            try:
                ans=ans.split(":")[1].strip(" \n*")
            except:
                if "YES" in ans or "yes" in ans or "Yes" in ans:
                    ans="YES"
                else:
                    ans="NO"
            if ans.startswith("YES"):
                GeneratedAnswers[mention].append("YES")
            else:
                GeneratedAnswers[mention].append("NO")

            with open(intermediatePath,"a+") as f:
                f.write("mention: "+mention)
                f.write("\n")
                f.write("candidate: "+candidate)
                f.write("\n")
                f.write("answer: "+ans)
                f.write("\n")
                f.write("-"*60)
                f.write("\n")
        
    return GeneratedAnswers

def run_prompt_with_score_wrapper(args):
    return run_prompt_with_score(*args)

def count_results(testData,generatedanswers,entityKb):
    Acc=[0]*4#true positive,False positive,true negative,false negative
    for line in testData:
        mention=line["mention"]
        labels=line["labels"]
        pred=generatedanswers[mention]
        golden_cui=line["golden_cui"]
        golden_entity=entityKb[golden_cui][0].lower()
        concepts=line["concepts"]
        candidates=line["candidates"]
        
        for i in range(len(candidates)):
            if labels[i]==1 and pred[i]=="YES":
                Acc[0]+=1
            elif labels[i]==0 and pred[i]=="YES":
                Acc[1]+=1
            elif labels[i]==0 and pred[i]=="NO":
                Acc[2]+=1
            elif labels[i]==0 and pred[i]=="YES":
                Acc[3]+=1
    return Acc

def main(args):
    test=load_data(args.testDataPath)
    testData=process_test_data(test)
    testDatas=split_data(testData)
    entityKb=load_dict(args.entityKbPath)
    prompt=load_prompt(args.promptPath,args.shot_number)
    apiKeys=load_keys(args.keyPath)
    # clients=create_clients(args.baseUrl,apiKeys)
    temperature=args.temperature
    maxTokens=args.maxTokens
    outputDir=args.outputDir
    intermediatePathDir=args.intermediatePath

    if not os.path.exists(intermediatePathDir):
        os.mkdir(intermediatePathDir)

    intermediatePaths=[intermediatePathDir+str(i)+".txt" for i in range(41)]
    pid=os.getpid()
    with open(intermediatePaths[40],"w") as file:
        file.write(f"testDataPath:{args.testDataPath}\n"+
                   f"entityKbPath:{args.entityKbPath}\n"+
                   f"promptPath:{args.promptPath}\n"+
                   f"shot_num:{args.shot_number}\n"+
                   f"temperature:{args.temperature}\n"+
                   f"maxTokens:{args.maxTokens}\n"+
                   f"outputDir:{args.outputDir}\n"+
                   f"intermediatePath:{args.intermediatePath}\n"
                   f"prompt:{prompt}\n"+
                   f"进程id:{pid}\n"
                   )
        file.write("_"*20)
        file.write("\n")
    
    mypool=multiprocessing.Pool(processes=40)
    argsList=[(prompt,testDatas[i],args.baseUrl,apiKeys[i],temperature,maxTokens,intermediatePaths[i]) for i in range(40)]

    # argsf=[(10),(20)]

    generatedAnswers=mypool.map(run_prompt_with_score_wrapper,argsList)
    # generatedAnswers=mypool.map(run_prompt_with_score,argsf)
    # generatedAnswers=run_prompt_with_score(prompt,testData,clients,temperature,maxTokens,intermediatePath)
    
    answers= {k: v for d in generatedAnswers for k, v in d.items()}
    results1=count_results(testData,answers,entityKb)
    print(results1)

    output={"answers":answers,"results1":results1}
    # #results1表示直接按照值和顺序进行排序，results2表示在此基础上考虑多个值预测为1的情况
    with open(outputDir,"w") as f:
        json.dump(output,f)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--testDataPath',
                        default="../../../datasets/ask_a_patient/small_500_0.json")
    parser.add_argument('--entityKbPath',
                        default="../../../datasets/ask_a_patient/entity_kb.json")
    parser.add_argument('--promptPath',
                        default="../prompts/parallel/positive/positive5.txt")
    parser.add_argument('--shot_number',type=int,
                        default=5)
    parser.add_argument('--baseUrl',
                        default="https://35api.huinong.co/v1")
    parser.add_argument('--keyPath',
                        default="../../../datasets/keys.json")
    parser.add_argument('--temperature',type=float,
                        default=0)
    parser.add_argument('--maxTokens',type=int,
                        default=128)
    parser.add_argument('--outputDir',
                        default="../results/judge/parallel/positive/score_5_five_shot.json")
    parser.add_argument('--intermediatePath',
                        default="../results/judge/records/")
    args = parser.parse_args()

    main(args)