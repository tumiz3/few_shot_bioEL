import json
import multiprocessing
import numpy as np
import os
import re
import argparse
from tqdm import tqdm
from copy import deepcopy
from openai import OpenAI

def load_keys(keyPath:str) -> (str,list):
    with open(keyPath,"r") as f:
        key=json.load(f)
    #key为一个字典，其中存储了baseUrl和keys，并且键为"baseUrl"和"keys"
    baseUrl=key["baseUrl"]
    keys=key["keys"]
    return baseUrl,keys

def load_data(testDataPath:str) -> dict:
    with open(testDataPath,"r") as f:
        testData=json.load(f)
    return testData

def split_data(data:list)->list:
    split_list = np.array_split(data, 20)
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
    systemRole=prompt.split("|split|")[0].strip()
    promptContent=prompt.split("|split|")[1].strip()
    others=prompt.split("|split|")[2].strip()
    prompts=promptContent.split("Example")
    # if "mix" in prompt:
    #     shot_number=2*shot_number
    # if "all" in prompt:
    #     shot_number=3*shot_number
    usedPrompt=prompts[:shot_number+1]
        
    newPrompt="Example".join(usedPrompt).strip()+"\n\n"+others

    return systemRole,newPrompt.strip()

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

def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

def contains_float(string):
    # 正则表达式匹配浮点数，包括整数、小数、带符号的数
    pattern = r'[-+]?\d*\.\d+|\d+'
    return bool(re.search(pattern, string))

def extract_floats(mention,string,candidate):
    # 正则表达式匹配浮点数，包括整数、小数、带符号的数
    pattern = r'[-+]?\d*\.\d+|\d+'
    scores=[match for match in re.findall(pattern, string)]

    length=20
    for score in scores:
        stringIndex=string.find(score)
        startIndex=max(0,stringIndex-length)
        endIndex=min(stringIndex+length,len(string))
        substring=string[startIndex:endIndex]

        if substring in mention or substring in candidate:
            continue

        return score

    return "0.0"


def create_client(base_url:str,api_key:str):
    client = OpenAI(base_url=base_url,api_key=api_key)
    return client

def ask(systemRole,prompt,client,model,temperature,max_tokens):
    response = client.chat.completions.create(
        model=model,#Currently points to gpt-3.5-turbo-0125,input max tokens:16385,output max tokens:196
        messages=[
            {"role":"system","content":systemRole},
            {"role":"user","content":prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    ans = response.choices[0].message.content
    return ans

def run_prompt_with_score(systemRole,manual_prompt:str,testData:list,base_url,key,model,temperature,max_tokens,intermediatePath):
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
            prompt=prompt.replace("entity",candidate)
            try:
                ans=ask(systemRole,prompt,client,model,temperature,max_tokens)
            except:
                continue
            # ans=ask(prompt,client,temperature,64)
            generatedOutput[mention].append(ans)
            # ans=ask(prompt,client,temperature,64)
            i+=1
            print(ans)
            print("\n")
            print("-"*60)
            # if ":" in ans:
            #     ans=ans.split(":")[1].strip(" \t,.，。\n")
            if not is_float(ans):
                    if contains_float(ans):
                        ans=extract_floats(mention,ans,candidate)
                    else:
                        ans="0.0"
            
            # print("\n")
            GeneratedAnswers[mention].append(eval(ans))

        with open(intermediatePath,"a+") as f:
            f.write(mention)
            f.write("\n")
            f.write(",".join(candidates))
            f.write("\n")
            f.write("\n".join(generatedOutput[mention]))
            f.write("\n")
            f.write("_"*60)
            f.write("\n")
        
    return GeneratedAnswers

def run_prompt_with_score_wrapper(args):
    return run_prompt_with_score(*args)

def count_results(testData,generatedanswers,entityKb):
    Acc=[0,0,0,0,0,0,0,0,0,0]
    Acc_parallel=[0] * 10
    for line in testData:
        mention=line["mention"]
        labels=line["labels"]
        pred=generatedanswers[mention]
        golden_cui=line["golden_cui"]
        golden_entity=entityKb[golden_cui][0].lower()
        concepts=line["concepts"]
        candidates=line["candidates"]
        sorted_indices = sorted(range(len(pred)), key=lambda x: pred[x], reverse=True)

        index_1=[i for i, x in enumerate(pred) if x == 1 or x==1.0]
        if index_1 !=[]:
            pred_candidates=[candidates[i] for i in index_1]
            if golden_entity in pred_candidates:
                Acc_parallel=[acc+1 for acc in Acc_parallel]
            else:
                for i in range(len(sorted_indices)):
                    if candidates[sorted_indices[i]]==golden_entity:
                        for j in range(i,10):
                            Acc_parallel[j]+=1
                        break
        else:
            for i in range(len(sorted_indices)):
                if candidates[sorted_indices[i]]==golden_entity:
                    for j in range(i,10):
                        Acc_parallel[j]+=1
                    break
        for i in range(len(sorted_indices)):
            if candidates[sorted_indices[i]]==golden_entity:
                for j in range(i,10):
                    Acc[j]+=1
                break
    print(Acc)
    return ([acc/len(testData) for acc in Acc],[acc/len(testData) for acc in Acc_parallel])

def main(args):
    test=load_data(args.testDataPath)
    testData=process_test_data(test)
    testDatas=split_data(testData)
    entityKb=load_dict(args.entityKbPath)
    systemRole,prompt=load_prompt(args.promptPath,args.shot_number)
    baseUrl,apiKeys=load_keys(args.keyPath)
    model=args.model
    # clients=create_clients(args.baseUrl,apiKeys)
    temperature=args.temperature
    maxTokens=args.maxTokens
    outputDir=args.outputDir
    intermediatePathDir=args.intermediatePath
    numberOfProcesses=args.k


    if not os.path.exists(intermediatePathDir):
        os.mkdir(intermediatePathDir)

    intermediatePaths=[intermediatePathDir+str(i)+".txt" for i in range(numberOfProcesses)]
    pid=os.getpid()
    with open(intermediatePaths[numberOfProcesses],"w") as file:
        file.write(f"testDataPath:{args.testDataPath}\n"+
                   f"entityKbPath:{args.entityKbPath}\n"+
                   f"promptPath:{args.promptPath}\n"+
                   f"shot_num:{args.shot_number}\n"+
                   f"temperature:{args.temperature}\n"+
                   f"maxTokens:{args.maxTokens}\n"+
                   f"outputDir:{args.outputDir}\n"+
                   f"intermediatePath:{args.intermediatePath}\n"
                   f"prompt:{prompt}\n"+
                   f"进程id:{pid}\n"+
                   f"模型：{model}"+
                   f"线程数：{numberOfProcesses}\n"
                   )
        file.write("_"*1)
        file.write("\n")
    
    mypool=multiprocessing.Pool(processes=numberOfProcesses)
    argsList=[(prompt,testDatas[i],baseUrl,apiKeys[i],model,temperature,maxTokens,intermediatePaths[i]) for i in range(numberOfProcesses)]

    # argsf=[(10),(1)]

    generatedAnswers=mypool.map(run_prompt_with_score_wrapper,argsList)
    # generatedAnswers=mypool.map(run_prompt_with_score,argsf)
    # generatedAnswers=run_prompt_with_score(prompt,testData,clients,temperature,maxTokens,intermediatePath)
    
    answers= {k: v for d in generatedAnswers for k, v in d.items()}
    results1,results2=count_results(testData,answers,entityKb)
    print(results1,results2)

    output={"answers":answers,"results1":results1,"results2":results2}
    #results1表示直接按照值和顺序进行排序，results2表示在此基础上考虑多个值预测为1的情况
    with open(outputDir,"w") as f:
        json.dump(output,f)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--testDataPath',
                        default="../datasets/ask_a_patient/test_sapbert_100.json")
    parser.add_argument('--entityKbPath',
                        default="../datasets/ask_a_patient/entity_kb.json")
    parser.add_argument('--promptPath',
                        default="./prompts_and_outputs/ask_a_patient/simple_prompt/prompt.txt")
    parser.add_argument('--shot_number',type=int,
                        default=5)
    parser.add_argument('--keyPath',
                        default="../keys/bianxieKeys.json")
    parser.add_argument('--model',
                        default="gpt-3.5-turbo")
    parser.add_argument('--temperature',type=float,
                        default=0.7)
    parser.add_argument('--maxTokens',type=int,
                        default=64)
    parser.add_argument('--k',
                        default=20)
    parser.add_argument('--outputDir',
                        default="./prompts_and_outputs/ask_a_patient/result")
    parser.add_argument('--intermediatePath',
                        default="./prompts_and_outputs/ask_a_patient/records/")
    args = parser.parse_args()

    main(args)