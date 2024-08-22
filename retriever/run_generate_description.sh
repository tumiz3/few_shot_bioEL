declare -A shots
shots=(
    ["one"]=1
    ["two"]=2
    ["three"]=3
    ["four"]=4
    ["five"]=5
)

dataset="ask_a_patient"

shot="five"
nohup python generate_description.py --testDataPath ../../initial_data/${dataset}/small_500_0.json \
                                --entityKbPath ../../initial_data/${dataset}/entity_kb.json \
                                --promptPath ../../designed_prompts/${dataset}/retriever/write_a_description.txt \
                                --shot_number ${shots[$shot]} \
                                --baseUrl https://35api.huinong.co/v1 \
                                --api_key sk-MXCbHJ2cpooin9Bq43Af80EcFe4f4a9cB0D4F385D3B5DeC6 \
                                --temperature 0.7 \
                                --maxTokens 64 \
                                --outputDir ../../retrieval_answers/${dataset}/answers/generate_description_entities_${shot}_shot.json \
                                --intermediatePath ../../retrieval_answers/${dataset}/records/generate_description_entitiess${shot}_shot.txt > "nohup_file/${dataset}_${shot}.txt" 2>&1 &