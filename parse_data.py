import pandas as pd
import os
import re

DEBUG = False

def parse(source_dir, network):
    filepath = os.path.join(source_dir, network, network + '.ratings')

    f = open(filepath, 'r')
    score = []
    summaries = []
    network_choice = []
    number = []

    for line in f:
        raw_data = line.split()
        name = raw_data[1]
        number.append(name)
        score.append(raw_data[3])
        summary_file = open(os.path.join(source_dir, network, name + "_summary.txt"), 'r', encoding='utf8')
        summary = summary_file.readline()
        summaries.append(summary)
        network_choice.append(network)
        if DEBUG:
            print(f"File: {name}\nScore: {raw_data[3]}\nSummary: {summary}\n\n")
        summary_file.close()
    f.close()
    
    df = pd.DataFrame(list(zip(number, summaries, score, network_choice)), columns=['Number', 'Summary', 'Score', 'Network'])

    if DEBUG:
        print(df)
    df.to_csv(os.path.join('.', 'CSVs', network + ".csv"), index=False)

    return df
    
def parseHTML(source_dir, network):
    filepath = os.path.join(source_dir, network, network + '.ratings')

    f = open(filepath, 'r')
    score = []
    summaries = []
    network_choice = []
    number = []

    for line in f:
        raw_data = line.split()
        name = raw_data[1]
        number.append(name)
        score.append(raw_data[3])
        summary_file = open(os.path.join(source_dir, network, "peer_" + name), 'r', encoding='utf8')
        summary = summary_file.readline()
        x = re.search("<.+>\\[.+\\]<.+> <.+>(.+)<.+>", summary)
        summary = x.group(1)
        summaries.append(summary)
        network_choice.append(network)
        if DEBUG:
            print(f"File: {name}\nScore: {raw_data[3]}\nSummary: {summary}\n\n")
        summary_file.close()
    f.close()

    df = pd.DataFrame(list(zip(number, summaries, score, network_choice)), columns=['Number', 'Summary', 'Score', 'Network'])

    if DEBUG:
        print(df)
    
    df.to_csv(os.path.join('.', 'CSVs', network + ".csv"), index=False)
    
    return df

if __name__ == "__main__":
    first = True
    networks = {}
    networks["Potsawee_Manakul"] = ['cued_speechUniv1', 'cued_speechUniv2', 'cued_speechUniv3', 'cued_speechUniv4']
    networks["Hannes_Karlbom" ] = ['hk_uu_podcast1']
    networks["Shadi_Rezapour"] = ['categoryaware1', 'categoryaware2', 'coarse2fine']
    networks["Sravana_Reddy"] = ['bartcnn', 'bartpodcasts', 'onemin', 'textranksegments', 'textranksentences']
    networks["Rachel_Zheng"] = ['udel_wang_zheng1', 'udel_wang_zheng2', 'udel_wang_zheng3','udel_wang_zheng4']
    networks["Paul_Owoicho"] = ['2306987O_abs_run1', '2306987O_extabs_run2', '2306987O_extabs_run3']
    networks["Kaiqiang_Song"] = ['UCF_NLP1', 'UCF_NLP2']
    networks["Sumanta_Kashyapi"] = ['unhtrema1', 'unhtrema2', 'unhtrema3', 'unhtrema4']

    df = pd.DataFrame()
    for key in networks:
        data_dir = os.path.join(".", "data", key)
        
        current_networks = networks[key]
        for network in current_networks:
            current_path = os.path.join(data_dir, network)
            temp = pd.DataFrame()
            if any(File.endswith(".txt") for File in os.listdir(current_path)):
                temp = parse(data_dir, network)
            else:
                temp = parseHTML(data_dir, network)
            
            df = df.append(temp, ignore_index=True)
    
    if (DEBUG):
        print(df)

    df.to_csv(os.path.join('.', 'CSVs', 'RatedSummaryComprehensive.csv'), index=False)
                

