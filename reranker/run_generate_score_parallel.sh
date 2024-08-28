declare -A shots
shots=(
    ["one"]=1
    ["two"]=2
    ["three"]=3
    ["four"]=4
    ["five"]=5
    ["six"]=6
)

dataset="smm4h"
testFileName="test_sapbert.json"

list1=("rule_2")
shot_number=("six")

for prompt in "${list1[@]}"
do
      for shot in "${shot_number[@]}"
      do
      nohup python generate_score_parallel.py --testDataPath ../datasets/${dataset}/${testFileName} \
                                      --entityKbPath ../datasets/${dataset}/entity_kb.json \
                                      --promptPath ./prompts_and_outputs/${dataset}/${prompt}/prompt.txt \
                                      --shot_number ${shots[$shot]} \
                                      --keyPath ../keys/bianxieKeys.json \
                                      --model gpt-4o-mini \
                                      --temperature 0 \
                                      --maxTokens 128 \
                                      --k 100 \
                                      --outputDir ./prompts_and_outputs/${dataset}/${prompt}/result.json \
                                      --intermediatePath ./prompts_and_outputs/${dataset}/${prompt}/records/ > "./prompts_and_outputs/${dataset}/${prompt}/nohup_record.txt" 2>&1 &
      done
done