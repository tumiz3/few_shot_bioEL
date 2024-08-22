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
list2=("4")
list3=("parallel")
shot="six"

for prompt in "${list3[@]}"
do
    for type in "${list1[@]}"
    do
        for number in "${list2[@]}"
        do
            nohup python generate_score_parallel_score.py --testDataPath ../../../datasets/${dataset}/small_500_0.json \
                                            --entityKbPath ../../../datasets/${dataset}/entity_kb.json \
                                            --promptPath ../prompts/${prompt}/${type}/${type}${number}.txt \
                                            --shot_number ${shots[$shot]} \
                                            --baseUrl https://35api.huinong.co/v1 \
                                            --keyPath ../../../datasets/keys.json \
                                            --temperature 0 \
                                            --maxTokens 128 \
                                            --outputDir ../results/judge/${prompt}/${type}/score_${number}_${shot}_shot.json \
                                            --intermediatePath ../results/judge/records/generate_score_${prompt}_${type}_${number}_${shot}_shot/ > "../nohup_file/${dataset}_${prompt}_${type}${number}_${shot}" 2>&1 &
        done
    done
done