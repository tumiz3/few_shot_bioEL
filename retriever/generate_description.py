import json
import os
import argparse
from tqdm import tqdm
from copy import deepcopy
from openai import OpenAI
def load_data(testDataPath:str) -> dict:
    with open(testDataPath,"r") as f:
        testData=json.load(f)
    return testData

def load_dict(entityKbPath:str) -> dict:
    with open(entityKbPath,"r") as f:
        entityKb=json.load(f)
    return entityKb

def load_prompt(promptPath:str,shot_number:int) -> str:
    #获取指定位置的prompt，同时根据shot_number选择prompt中例子的数量
    with open(promptPath,"r") as f:
        prompt=f.read()
    
    prompts=prompt.split("Example")
    usedPrompt=prompts[:shot_number+1]
        
    newPrompt="Example".join(usedPrompt).strip()

    return newPrompt

def process_test_data(test:dict) -> list:
    testData=[]
    for item in test["queries"]:
        mention=item["mentions"][0]["mention"]
        candidatesData=item["mentions"][0]["candidates"]
        candidates=[candidate["name"] for candidate in candidatesData]
        labels=[candidate["label"] for candidate in candidatesData]
        testData.append({"mention":mention,"candidates":candidates,"labels":labels})
    return testData

def create_client(base_url:str,api_key:str):
    client = OpenAI(base_url=base_url,api_key=api_key)
    return client

def ask(prompt,client,temperature,max_tokens):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",#Currently points to gpt-3.5-turbo-0125,input max tokens:16385,output max tokens:4096
        messages=[
            {"role":"system","content":"As a professional doctor, you can understandard the meaning of a medical term and write a paragraph that accurately describes the medical term."},
            {"role":"user","content":prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    ans = response.choices[0].message.content
    return ans

def run_prompt_with_score(manual_prompt:str,testData:list,client,temperature,max_tokens,intermediatePath):
    GeneratedAnswers={}
    for sample in tqdm(testData.keys()):
        mention=testData[sample][0]
        GeneratedAnswers[mention]=""
        
        for numberDescription in range(1):            
            prompt=manual_prompt       
            prompt=prompt.replace("mention",mention)
            with open(intermediatePath,"a+") as f:
                ans=ask(prompt,client,temperature,max_tokens)
                f.write(f"mention:{mention}")
                f.write(ans)
                f.write("\n")
            GeneratedAnswers[mention]+=ans
            # GeneratedAnswers[mention]+="__"
    return GeneratedAnswers

def main(args):
    testData=process_test_data(load_data(args.testDataPath))
    entityKb=load_dict(args.entityKbPath)
    prompt=load_prompt(args.promptPath,args.shot_number)
    client=create_client(args.baseUrl,args.api_key)
    temperature=args.temperature
    maxTokens=args.maxTokens
    outputDir=args.outputDir
    intermediatePath=args.intermediatePath
    pid=os.getpid()
    with open(intermediatePath,"w") as file:
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
        
    generatedAnswers=run_prompt_with_score(prompt,entityKb,client,temperature,maxTokens,intermediatePath)
    with open(outputDir,"w") as f:
        json.dump(generatedAnswers,f)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--testDataPath',
                        default="../../initial_data/ask_a_patient/small_1000_0.json")
    parser.add_argument('--entityKbPath',
                        default="../../initial_data/ask_a_patient/entity_kb.json")
    parser.add_argument('--promptPath',
                        default="../../designed_prompts/ask_a_patient/retriever/write_a_description.txt")
    parser.add_argument('--shot_number',type=int,
                        default=5)
    parser.add_argument('--baseUrl',
                        default="https://35api.huinong.co/v1")
    parser.add_argument('--api_key',
                        default="sk-AteooaWbEoDABZdK0a7e32904d38497d888f7399378cA825")
    parser.add_argument('--temperature',type=float,
                        default=0.7)
    parser.add_argument('--maxTokens',type=int,
                        default=64)
    parser.add_argument('--outputDir',
                        default="../../retrieval_answers/ask_a_patient/answers/generate_description_five_shot.json")
    parser.add_argument('--intermediatePath',
                        default="../../retrieval_answers/ask_a_patient/records/generate_description_five_shot.txt")
    args = parser.parse_args()

    main(args)
    