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
        testData=[json.loads(line) for line in f]
    return testData


def split_data(data: list,numberOfProcesses) -> list:
    split_list = np.array_split(data, numberOfProcesses)
    split_list = [arr.tolist() for arr in split_list]
    return split_list

def split_entity_kb(entityKb,numberOfProcesses,tryTest) -> list:
    if tryTest:
        keys=list(entityKb.keys())
    else:
        keys = list(entityKb.keys())
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


def load_prompt(promptPath: str, shot_number: int):
    #获取指定位置的prompt，同时根据shot_number选择prompt中例子的数量
    with open(promptPath, "r") as f:
        prompt = f.read()
    systemRole = prompt.split("|split|")[0].strip()
    promptContent = prompt.split("|split|")[1].strip()
    if len(prompt.split("|split|"))>2:
        others = prompt.split("|split|")[2].strip()
    else:
        others=""
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
        # type=item["mention_data"][0]["type"]
        testData.append({"mention":mention,"golden_cui":golden_cui,"text":text})
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
        mention=sample["mention"]
        if mention in GeneratedAnswers.keys():
            continue
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
            if numberOfRounds != 1:
                GeneratedAnswers[mention] += "__"
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
                    if numberOfRounds != 1:
                        GeneratedAnswers[entity] += "__"
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
                if numberOfRounds != 1:
                    GeneratedAnswers[entity]+="__"
    return GeneratedAnswers

def generateDescriptionForMentionWrapper(args):
    return generateDescriptionForMention(*args)

def generateDescriptionForEntityWrapper(args):
    return generateDescriptionForEntityKb(*args)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):

    tryTest=args.tryTest
    if tryTest:
        testData = process_test_data(load_data(args.testDataPath)[:20])
    else:
        testData=process_test_data(load_data(args.testDataPath))
    numberOfProcesses = args.numberOfProcesses
    testDatas=split_data(testData,numberOfProcesses)
    entityKb=load_dict(args.entityKbPath)
    ids=split_entity_kb(entityKb, numberOfProcesses,tryTest)
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

    mentionIntermediatePaths = [mentionIntermediateDir + str(i) + ".txt" for i in range(numberOfProcesses + 1)]
    entityIntermediatePaths = [entityIntermediateDir+ str(i) + ".txt" for i in range(numberOfProcesses + 1)]
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
        (systemRole,prompt,entityKb,ids[i],model,temperature,maxTokens, entityIntermediatePaths[i],numberOfRounds, useSynonyms,baseUrl,keys[i]) for
         i in range(numberOfProcesses)
    ]

    # argsf=[(10),(1)]

    generatedAnswersMention = mypool.map(generateDescriptionForMentionWrapper, argsListMention)
    generatedAnswersMention = {k: v for d in generatedAnswersMention for k, v in d.items()}
    with open(mentionDescriptionDir,"w") as f:
        json.dump(generatedAnswersMention,f)

    if generateEntityDescription:
        generatedAnswersEntity = mypool.map(generateDescriptionForEntityWrapper, argListEntity)
        generatedAnswersEntity = {k: v for d in generatedAnswersEntity for k, v in d.items()}
        with open(entityDescriptionDir,"w") as f:
            json.dump(generatedAnswersEntity,f)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--testDataPath',
                        default="../datasets/ask_a_patient/test.json")
    parser.add_argument('--entityKbPath',
                        default="../datasets/ask_a_patient/entity_kb.json")
    parser.add_argument('--promptPath',
                        default="./prompts_and_outputs/ask_a_patient/simple_prompt/prompt.txt")
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
                        default=1)
    parser.add_argument('--generateEntityDescription', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--useSynonyms', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--tryTest', type=str2bool, nargs='?', const=True, default=False, help="Boolean flag")
    parser.add_argument('--mentionOutputDir',
                        default="./prompts_and_outputs/ask_a_patient/simple_prompt/mentionDescription.json")
    parser.add_argument('--entityOutputDir',
                        default="./prompts_and_outputs/ask_a_patient/simple_prompt/entityDescription.json")
    parser.add_argument('--mentionIntermediatePath',
                        default="./prompts_and_outputs/ask_a_patient/simple_prompt/mentionRecords/")
    parser.add_argument('--entityIntermediatePath',
                        default="./prompts_and_outputs/ask_a_patient/entityRecords/")
    args = parser.parse_args()

    main(args)
    