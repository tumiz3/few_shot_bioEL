declare -A shots
shots=(
    ["one"]=1
    ["two"]=2
    ["three"]=3
    ["four"]=4
    ["five"]=5
    ["six"]=6
)


dataset="ask_a_patient"
testFileName="test_sapbert_100.json"

list1=("simple_prompt")
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
                                      --model gpt-3.5-turbo \
                                      --temperature 0 \
                                      --maxTokens 128 \
                                      --k 20 \
                                      --outputDir ./prompts_and_outputs/${dataset}/${prompt}/result.json \
                                      --intermediatePath ./prompts_and_outputs/${dataset}/${prompt}/records/ > "./prompts_and_outputs/${dataset}/${prompt}/nohup_record.txt" 2>&1 &
      done
done