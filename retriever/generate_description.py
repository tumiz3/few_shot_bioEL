import json
import os
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing
from copy import deepcopy
from openai import OpenAI


def load_data(testDataPath:str) -> list:
    with open(testDataPath,"r") as f:
        testData=json.load(f)
    return testData


def split_data(data: list,numberOfProcesses) -> list:
    split_list = np.array_split(data, numberOfProcesses)
    split_list = [arr.tolist() for arr in split_list]
    return split_list

def split_entity_kb(entityKb,numberOfProcesses) -> list:
    keys=list(entityKb.keys())
    split_list = np.array_split(keys, numberOfProcesses)
    split_list = [arr.tolist() for arr in split_list]
    return split_list


def load_dict(entityKbPath:str) -> dict:
    with open(entityKbPath,"r") as f:
        entityKb=json.load(f)
    return entityKb


def load_keys(keyPath: str) -> (str, list):
    with open(keyPath, "r") as f:
        key = json.load(f)
    #key为一个字典，其中存储了baseUrl和keys，并且键为"baseUrl"和"keys"
    baseUrl = key["baseUrl"]
    keys = key["keys"]
    return baseUrl, keys


def load_prompt(promptPath: str, shot_number: int) -> tuple[str, str]:
    #获取指定位置的prompt，同时根据shot_number选择prompt中例子的数量
    with open(promptPath, "r") as f:
        prompt = f.read()
    systemRole = prompt.split("|split|")[0].strip()
    promptContent = prompt.split("|split|")[1].strip()
    others = prompt.split("|split|")[2].strip()
    prompts = promptContent.split("Example")
    # if "mix" in prompt:
    #     shot_number=2*shot_number
    # if "all" in prompt:
    #     shot_number=3*shot_number
    usedPrompt = prompts[:shot_number + 1]

    newPrompt = "Example".join(usedPrompt).strip() + "\n\n" + others

    return systemRole, newPrompt.strip()

def process_test_data(test:list) -> list:
    testData=[]
    for item in test:
        mention=item["mention_data"][0]["mention"]
        text=item["text"].replace("[E1]","").replace("[/E1]","")
        golden_cui=item["mention_data"][0]["kb_id"]
        type=item["mention_data"][0]["type"]
        testData.append({"mention":mention,"golden_cui":golden_cui,"type":type,"text":text})
    return testData


def create_client(base_url:str,api_key:str):
    client = OpenAI(base_url=base_url,api_key=api_key)
    return client

def ask(systemRole, prompt, client, model, temperature, max_tokens):
    response = client.chat.completions.create(
        model=model,#Currently points to gpt-3.5-turbo-0125,input max tokens:16385,output max tokens:4096
        messages=[
            {"role":"system","content":systemRole},
            {"role":"user","content":prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    ans = response.choices[0].message.content
    return ans

def generateDescriptionForMention(systemRole: str, manual_prompt:str, testData:list, model, temperature,max_tokens,intermediatePath,numberOfRounds,baseUrl,api_key):
    client=create_client(baseUrl,api_key)
    GeneratedAnswers={}
    for sample in tqdm(testData):
        mention=testData[sample][0]
        GeneratedAnswers[mention]=""
        
        for numberDescription in range(numberOfRounds):
            prompt=manual_prompt       
            prompt=prompt.replace("mention",mention)
            with open(intermediatePath,"a+") as f:
                ans=ask(systemRole,prompt,client,model,temperature,max_tokens)
                f.write(f"mention:{mention}")
                f.write("\n")
                f.write(ans)
                f.write("\n\n")
            GeneratedAnswers[mention]+=ans
            # GeneratedAnswers[mention]+="__"
    return GeneratedAnswers


def generateDescriptionForEntityKb(systemRole: str, manual_prompt: str, entityKb: dict, ids:list, model, temperature, max_tokens,
                          intermediatePath, numberOfRounds,useSynonyms: bool,baseUrl,api_key):
    client=create_client(baseUrl,api_key)
    GeneratedAnswers = {}
    for id in tqdm(ids):
        if useSynonyms:
            for entity in entityKb[id]:
                GeneratedAnswers[entity] = ""
                for numberDescription in range(numberOfRounds):
                    prompt = manual_prompt
                    prompt = prompt.replace("mention", entity)
                    with open(intermediatePath, "a+") as f:
                        ans = ask(systemRole, prompt, client, model, temperature, max_tokens)
                        f.write(f"entity:{entity}")
                        f.write("\n")
                        f.write(ans)
                        f.write("\n\n")
                    GeneratedAnswers[entity] += ans
        else:
            entity = entityKb[id][0]
            GeneratedAnswers[entity] = ""

            for numberDescription in range(numberOfRounds):
                prompt = manual_prompt
                prompt = prompt.replace("mention", entity)#prompt中需要替换的是mention，无论是用于生成mention的描述还是entity的描述
                with open(intermediatePath, "a+") as f:
                    ans = ask(systemRole, prompt, client, model, temperature, max_tokens)
                    f.write(f"mention:{entity}")
                    f.write("\n")
                    f.write(ans)
                    f.write("\n\n")
                GeneratedAnswers[entity] += ans
            # GeneratedAnswers[mention]+="__"
    return GeneratedAnswers

def generateDescriptionForMentionWrapper(args):
    return generateDescriptionForMention(*args)

def generateDescriptionForEntityWrapper(args):
    return generateDescriptionForEntityKb(*args)

def main(args):
    testData=process_test_data(load_data(args.testDataPath))
    numberOfProcesses = args.k
    testDatas=split_data(testData,numberOfProcesses)
    entityKb=load_dict(args.entityKbPath)
    ids=split_entity_kb(entityKb, numberOfProcesses)
    systemRole,prompt=load_prompt(args.promptPath,args.shot_number)
    baseUrl,keys=load_keys(args.keyPath)
    temperature=args.temperature
    maxTokens=args.maxTokens
    model=args.model
    numberOfRounds=args.numberOfRounds
    generateEntityDescription=args.generateEntityDescription
    useSynonyms=args.useSynonyms
    mentionDescriptionDir=args.mentionOutputDir
    entityDescriptionDir=args.entityOutputDir
    mentionIntermediateDir=args.mentionIntermediatePath
    entityIntermediateDir=args.entityIntermediatePath

    if not os.path.exists(mentionIntermediateDir):
        os.mkdir(mentionIntermediateDir)

    if not os.path.exists(entityIntermediateDir):
        os.mkdir(entityIntermediateDir)

    mentionIntermediatePaths = [mentionDescriptionDir + str(i) + ".txt" for i in range(numberOfProcesses + 1)]
    entityIntermediatePaths = [entityDescriptionDir + str(i) + ".txt" for i in range(numberOfProcesses + 1)]
    pid = os.getpid()
    with open(mentionIntermediatePaths[numberOfProcesses], "w") as file:
        file.write(f"testDataPath:{args.testDataPath}\n" +
                   f"entityKbPath:{args.entityKbPath}\n" +
                   f"promptPath:{args.promptPath}\n" +
                   f"shot_num:{args.shot_number}\n" +
                   f"temperature:{args.temperature}\n" +
                   f"maxTokens:{args.maxTokens}\n" +
                   f"outputDir:{mentionDescriptionDir}\n" +
                   f"intermediatePath:{mentionIntermediateDir}\n"
                   f"prompt:{prompt}\n" +
                   f"进程id:{pid}\n" +
                   f"模型：{model}\n" +
                   f"线程数：{numberOfProcesses}\n"+
                   f"重复次数:{numberOfRounds}\n"+
                   f"为实体生成描述：{generateEntityDescription}\n"+
                   f"使用实体的同义词：{useSynonyms}\n"
                   )
        file.write("_" * 1)
        file.write("\n")

    with open(entityIntermediatePaths[numberOfProcesses], "w") as file:
        file.write(f"testDataPath:{args.testDataPath}\n" +
                   f"entityKbPath:{args.entityKbPath}\n" +
                   f"promptPath:{args.promptPath}\n" +
                   f"shot_num:{args.shot_number}\n" +
                   f"temperature:{args.temperature}\n" +
                   f"maxTokens:{args.maxTokens}\n" +
                   f"outputDir:{entityDescriptionDir}\n" +
                   f"intermediatePath:{entityIntermediateDir}\n"
                   f"prompt:{prompt}\n" +
                   f"进程id:{pid}\n" +
                   f"模型：{model}" +
                   f"线程数：{numberOfProcesses}\n"+
                   f"重复次数:{numberOfRounds}\n" +
                   f"为实体生成描述：{generateEntityDescription}\n" +
                   f"使用实体的同义词：{useSynonyms}\n"
                   )
        file.write("_" * 1)
        file.write("\n")

    mypool = multiprocessing.Pool(processes=numberOfProcesses)
    argsListMention = [
        (systemRole, prompt, testDatas[i], model, temperature, maxTokens, mentionIntermediatePaths[i],numberOfRounds,baseUrl, keys[i]) for
        i in range(numberOfProcesses)]

    argListEntity=[
        (systemRole,prompt,entityKb,ids[i],model,temperature,maxTokens, entityIntermediatePaths[i],numberOfRounds, useSynonyms,baseUrl,keys[i] for
         i in range(numberOfProcesses))
    ]

    # argsf=[(10),(1)]

    generatedAnswersMention = mypool.map(generateDescriptionForEntityWrapper(), argsListMention)
    with open(mentionDescriptionDir,"w") as f:
        json.dump(generatedAnswersMention,f)

    if generateEntityDescription:
        generatedAnswersEntity = mypool.map(generateDescriptionForEntityWrapper(), argListEntity)
        with open(entityDescriptionDir,"w") as f:
            json.dump(generatedAnswersEntity,f)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--testDataPath',
                        default="../../initial_data/ask_a_patient/small_1000_0.json")
    parser.add_argument('--entityKbPath',
                        default="../../initial_data/ask_a_patient/entity_kb.json")
    parser.add_argument('--promptPath',
                        default="../../designed_prompts/ask_a_patient/retriever/write_a_description.txt")
    parser.add_argument('--numberOfProcesses', type=int,
                        default=20)
    parser.add_argument('--shot_number',type=int,
                        default=5)
    parser.add_argument('--keyPath',
                        default="../keys/bianxieKeys.json")
    parser.add_argument('--temperature',type=float,
                        default=0.7)
    parser.add_argument('--maxTokens',type=int,
                        default=64)
    parser.add_argument('--model', type=str,
                        default="gpt-3.5-turbo")
    parser.add_argument('--numberOfRounds', type=int,
                        default=2)
    parser.add_argument('--generateEntityDescription', type=bool, default=True),
    parser.add_argument('--useSynonyms', type=bool, default=False),
    parser.add_argument('--mentionOutputDir',
                        default="../../retrieval_answers/ask_a_patient/answers/generate_description_five_shot.json")
    parser.add_argument('--entityOutputDir',
                        default="../../retrieval_answers/ask_a_patient/answers/generate_description_five_shot.json")
    parser.add_argument('--mentionIntermediatePath',
                        default="../../retrieval_answers/ask_a_patient/records/generate_description_five_shot.txt")
    parser.add_argument('--entityIntermediatePath',
                        default="../../retrieval_answers/ask_a_patient/records/generate_description_five_shot.txt")
    args = parser.parse_args()

    main(args)
    