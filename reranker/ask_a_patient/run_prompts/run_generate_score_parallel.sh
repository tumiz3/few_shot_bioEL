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
list1=("positive")
list2=("7")
list3=("parallel")
shot_number=("six")

for prompt in "${list3[@]}"
do
    for type in "${list1[@]}"
    do
        for number in "${list2[@]}"
        do
            for shot in "${shot_number[@]}"
            do
            nohup python generate_score_parallel.py --testDataPath ../../initial_data/${dataset}/rule_test.json \
                                            --entityKbPath ../../initial_data/${dataset}/entity_kb.json \
                                            --promptPath ../../designed_prompts/${dataset}/re_ranker/${prompt}/${type}/${type}${number}.txt \
                                            --shot_number ${shots[$shot]} \
                                            --baseUrl https://4api.huinong.co/v1 \
                                            --keyPath ../../initial_data/keys.json \
                                            --temperature 0 \
                                            --maxTokens 128 \
                                            --outputDir ../../output/${dataset}/${prompt}/${type}/score_rule_${number}_${shot}_shot.json \
                                            --intermediatePath ../../output/${dataset}/records/generate_score_rule_${prompt}_${type}_${number}_${shot}_shot/ > "nohup_file/${dataset}_${prompt}_${type}${number}_${shot}" 2>&1 &
            done
        done
    done
done